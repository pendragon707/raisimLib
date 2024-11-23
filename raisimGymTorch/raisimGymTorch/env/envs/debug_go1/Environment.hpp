//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <deque>
#include <stdlib.h>
#include <math.h>
#include <cstdint>
#include <set>
#include <vector>
#include <string>
#include <random>
#include "../../RaisimGymEnv.hpp"
#include <fenv.h>
#include <stdexcept>

template <typename T>
inline void scaling(T &n, const T &max_src, const T &max_dst)
{
  n = n / max_src * max_dst;
}

inline void quatToEuler(const raisim::Vec<4> &quat, Eigen::Vector3d &eulerVec)
{
  double qw = quat[0], qx = quat[1], qy = quat[2], qz = quat[3];
  // roll (x-axis rotation)
  double sinr_cosp = 2 * (qw * qx + qy * qz);
  double cosr_cosp = 1 - 2 * (qx * qx + qy * qy);
  eulerVec[0] = std::atan2(sinr_cosp, cosr_cosp);

  // pitch (y-axis rotation)
  double sinp = 2 * (qw * qy - qz * qx);
  if (std::abs(sinp) >= 1)
    eulerVec[1] = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
  else
    eulerVec[1] = std::asin(sinp);

  // yaw (z-axis rotation)
  double siny_cosp = 2 * (qw * qz + qx * qy);
  double cosy_cosp = 1 - 2 * (qy * qy + qz * qz);
  eulerVec[2] = std::atan2(siny_cosp, cosy_cosp);
}

inline void eulerToRot(const Eigen::Vector3d &eulerVec, Eigen::Matrix3d &rot)
{
  double roll = eulerVec[0], pitch = eulerVec[1], yaw = eulerVec[2];
  Eigen::Quaterniond q;
  q = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
  rot = q.matrix();
}

namespace raisim
{

  class ENVIRONMENT : public RaisimGymEnv
  {

  public:
    explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable, int env_id) : RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable)
    {
      setSeed(env_id);
      for (int i = 0; i < 10; i++) {
        Eigen::VectorXd::Random(1)[0];
      }

      /// add objects
      READ_YAML(int, baseDim, cfg["baseDim"])
      READ_YAML(int, history_len, cfg["history_len"])
      
      /// Target Speed
      READ_YAML(double, pid_coeff, cfg["pid_coeff"])

      go1_ = world_->addArticulatedSystem(resourceDir_ + "/go1/urdf/go1.urdf");
      go1_->setName("go1");
      go1_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

      nFoot = 4;

      /// get robot data
      gcDim_ = go1_->getGeneralizedCoordinateDim();
      gvDim_ = go1_->getDOF();
      nJoints_ = gvDim_ - 6;
      base_mass_list = go1_->getMass();

      /// initialize containers
      gc_.setZero(gcDim_);
      gc_init_.setZero(gcDim_);
      gv_.setZero(gvDim_);
      gv_init_.setZero(gvDim_);
      pTarget_.setZero(gcDim_);
      vTarget_.setZero(gvDim_);
      pTarget12_.setZero(nJoints_);
      
      jt_mean_pos.setZero(nJoints_);
      jt_mean_pos << 0.05, 0.8, -1.4, -0.05, 0.8, -1.4, 0.05, 0.8, -1.4, -0.05, 0.8, -1.4;
      
      // randomize terrains
      gc_init_ << 0, 0, 0.44, 1.0, 0.0, 0.0, 0.0, jt_mean_pos;
      std::cout << "Gc init: " << gc_init_ << std::endl;

      /// set pd gains
      // std::cout << "Pid coeffs " <<  pid_coeff << std::endl;
      pgain = pid_coeff;
      dgain = 0.6;
      std::cout << "pgain: " <<  pgain << " dgain: " << dgain << std::endl;
      Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
      jointPgain.setZero();
      jointPgain.tail(nJoints_).setConstant(pgain);
      jointDgain.setZero();
      jointDgain.tail(nJoints_).setConstant(dgain);
      go1_->setPdGains(jointPgain, jointDgain);
      go1_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

      /// MUST BE DONE FOR ALL ENVIRONMENTS
      base_obDim_ = baseDim + /* privinfo */ (20 + 3 * (int)1 /*use_priv_vel*/ + nFoot * (int)1 /*use_slope_dots*/ + 1); // added step
      obDim_ = baseDim * history_len + base_obDim_;
      actionDim_ = nJoints_;
      actionMean_.setZero(actionDim_);
      actionStd_.setZero(actionDim_);
      obDouble_.setZero(base_obDim_);
      ob_delay.setZero(base_obDim_);
      ob_concat.setZero(obDim_);

      /// action & observation scaling
      actionMean_ = gc_init_.tail(nJoints_);
      std::cout << "actionMean_: " << actionMean_ << std::endl;
      double act_std_val = 0.4;
      actionStd_.setConstant(act_std_val);
      
      /// visualize if it is the first environment
      if (visualizable_)
      {
        server_ = std::make_unique<raisim::RaisimServer>(world_.get());
        server_->launchServer();
        server_->focusOn(go1_);
      }

      world_->addGround();
    }

    void init() final {}

    void kill_server()
    {
      server_->killServer();
    }

    void randomize_sim_params()
    {
      friction = 0.8;
      world_->setDefaultMaterial(friction, 0.0, 0.0);
      
      pgain = pid_coeff;
      dgain = 0.6;
      Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
      jointPgain.setZero();
      jointPgain.tail(nJoints_).setConstant(pgain);
      jointDgain.setZero();
      jointDgain.tail(nJoints_).setConstant(dgain);
      go1_->setPdGains(jointPgain, jointDgain);
    
      mass_params << 0.5, 0, 0;
    }

    void reset(bool resample) final
    {
      randomize_sim_params();
      gv_init_.setZero(gvDim_);
      gc_init_[2] = 0.4;
      go1_->setState(gc_init_, gv_init_);

      obs_history.clear();
      act_history.clear();
      
      // get to rest pos
      pTarget_.tail(nJoints_) = Eigen::VectorXd::Zero(12) + actionMean_;
      
      go1_->setPdTarget(pTarget_, vTarget_);

      for (int i = 0; i < 50; i++)
      {
        if (server_)
          server_->lockVisualizationServerMutex();
        world_->integrate();
        if (server_)
          server_->unlockVisualizationServerMutex();
      }

      for (int j = 0; j < 50; j++)
        act_history.push_back(Eigen::VectorXd::Zero(12));
      
      for (int j = 0; j < 100; j++)
        updateObservation();
      updateObservation();
      updateObservation();
      updateObservation();
    }

    void setSeed(int seed)
    {
      // seed_ = seed;
      std::srand(seed);
      for (int i = 0; i < 10; i++) {
        Eigen::VectorXd::Random(1)[0];
      }
    }

    float step(const Eigen::Ref<EigenVec> &action_vec) final
    {
      act_history.push_back(action_vec.cast<double>());
      auto action = act_history[act_history.size() - 3];

      // mean std normalize action
      pTarget12_ = action;
      pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
      pTarget12_ += actionMean_;
      pTarget_.tail(nJoints_) = pTarget12_;

      go1_->setPdTarget(pTarget_, vTarget_);

      for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++)
      {
        if (server_)
          server_->lockVisualizationServerMutex();
        world_->integrate();
        if (server_)
          server_->unlockVisualizationServerMutex();
      }
      
      updateObservation();
      return 0.0;
    }

    void updateObservation()
    {
      go1_->getState(gc_, gv_);
      raisim::Vec<4> quat;
      raisim::Mat<3, 3> rot;
      quat[0] = gc_[3];
      quat[1] = gc_[4];
      quat[2] = gc_[5];
      quat[3] = gc_[6];
      
      quatToEuler(quat, bodyOrientation_);
      obDouble_ << bodyOrientation_.head(2),
          gc_.tail(12), /// joint angles
          gv_.tail(12),
          act_history[act_history.size() - 1],
          0.0, 0.0, 0.0, //speed_vec,
          0.0, //(double)includeGRF * grf_bin_obs,
          0.0, //mass_params,
	        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,//motor_strength,
          0.0, //friction,
          0.0, 0.0, 0.0, //avgXYVel, avgYawVel,
          0.0, 0.0, 0.0, 0.0, //slope_dots,
          0.0; //isSlope;

      obs_history.push_back(obDouble_);
    }

    void observe(Eigen::Ref<EigenVec> ob) final
    {
      int lag = 1;
      int vec_size = obs_history.size();
      ob_delay << obs_history[vec_size - lag]; //, obs_history[obs_history.size() - 2], obs_history[obs_history.size() - 3];
      for (int i = 0; i < history_len; i++)
      {
        ob_concat.segment(baseDim * i, baseDim) << obs_history[vec_size - lag - history_len + i].head(baseDim);
      }
      ob_concat.tail(base_obDim_) << ob_delay;
      ob = ob_concat.cast<float>();
    }

    bool isTerminalState(float &terminalReward) final {
      return false;
    }

  private:
    int gcDim_, gvDim_, nJoints_;
    bool visualizable_ = false;
    raisim::ArticulatedSystem *go1_;
    Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
    double friction = 0.8;
    
    Eigen::VectorXd actionMean_, actionStd_, obDouble_, ob_delay;
    Eigen::Vector3d bodyOrientation_;
    
    Eigen::Vector3d mass_params;
    Eigen::VectorXd jt_mean_pos;
    
    double pgain = 50.;
    double dgain = 0.6;
    
    int nFoot = 0;
    
    std::vector<double> base_mass_list;
    
    std::deque<Eigen::VectorXd> obs_history;
    std::deque<Eigen::VectorXd> act_history;
    
    int baseDim;
    double pid_coeff = 55;

    int base_obDim_;
    Eigen::VectorXd ob_concat;
    int history_len;
  };
}
