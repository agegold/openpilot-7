import math
import numpy as np

from cereal import log
from common.filter_simple import FirstOrderFilter
from common.numpy_fast import clip, interp
from common.realtime import DT_CTRL
from selfdrive.controls.lib.latcontrol import LatControl, MIN_STEER_SPEED
from common.params import Params
from decimal import Decimal


class LatControlINDI(LatControl):
  def __init__(self, CP, CI):
    super().__init__(CP, CI)
    self.angle_steers_des = 0.

    A = np.array([[1.0, DT_CTRL, 0.0],
                  [0.0, 1.0, DT_CTRL],
                  [0.0, 0.0, 1.0]])
    C = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0]])

    # Q = np.matrix([[1e-2, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 10.0]])
    # R = np.matrix([[1e-2, 0.0], [0.0, 1e3]])

    # (x, l, K) = control.dare(np.transpose(A), np.transpose(C), Q, R)
    # K = np.transpose(K)
    K = np.array([[7.30262179e-01, 2.07003658e-04],
                  [7.29394177e+00, 1.39159419e-02],
                  [1.71022442e+01, 3.38495381e-02]])

    self.speed = 0.

    self.K = K
    self.A_K = A - np.dot(K, C)
    self.x = np.array([[0.], [0.], [0.]])

    self.mpc_frame = 0
    self.params = Params()

    self._RC = (CP.lateralTuning.indi.timeConstantBP, CP.lateralTuning.indi.timeConstantV)
    self._G = (CP.lateralTuning.indi.actuatorEffectivenessBP, CP.lateralTuning.indi.actuatorEffectivenessV)
    self._outer_loop_gain = (CP.lateralTuning.indi.outerLoopGainBP, CP.lateralTuning.indi.outerLoopGainV)
    self._inner_loop_gain = (CP.lateralTuning.indi.innerLoopGainBP, CP.lateralTuning.indi.innerLoopGainV)

    self.RC = 0
    self.G = 0
    self.outer_loop_gain = 0
    self.inner_loop_gain = 0

    self.steer_filter = FirstOrderFilter(0., self.RC, DT_CTRL)

    self.live_tune_enabled = False

    self.reset()

    self.li_timer = 0

  def reset(self):
    super().reset()
    self.steer_filter.x = 0.
    self.speed = 0.

  def live_tune(self, CP):
    self.mpc_frame += 1
    if self.mpc_frame % 300 == 0:
      self.outerLoopGain = float(Decimal(self.params.get("OuterLoopGain", encoding="utf8")) * Decimal('0.1'))
      self.innerLoopGain = float(Decimal(self.params.get("InnerLoopGain", encoding="utf8")) * Decimal('0.1'))
      self.timeConstant = float(Decimal(self.params.get("TimeConstant", encoding="utf8")) * Decimal('0.1'))
      self.actuatorEffectiveness = float(Decimal(self.params.get("ActuatorEffectiveness", encoding="utf8")) * Decimal('0.1'))
      self.RC = interp(self.speed, [0.], [self.timeConstant]) 
      self.G = interp(self.speed, [0.], [self.actuatorEffectiveness])
      self.outer_loop_gain = interp(self.speed, [0.], [self.outerLoopGain])
      self.inner_loop_gain = interp(self.speed, [0.], [self.innerLoopGain])
        
      self.mpc_frame = 0

  def update(self, active, CS, CP, VM, params, last_actuators, desired_curvature, desired_curvature_rate):
    self.speed = CS.vEgo

    self.RC = interp(self.speed, self._RC[0], self._RC[1])
    self.G = interp(self.speed, self._G[0], self._G[1])
    self.outer_loop_gain = interp(self.speed, self._outer_loop_gain[0], self._outer_loop_gain[1])
    self.inner_loop_gain = interp(self.speed, self._inner_loop_gain[0], self._inner_loop_gain[1])

    self.li_timer += 1
    if self.li_timer > 100:
      self.li_timer = 0
      self.live_tune_enabled = self.params.get_bool("OpkrLiveTunePanelEnable")
    if self.live_tune_enabled:
      self.live_tune(CP)

    # Update Kalman filter
    y = np.array([[math.radians(CS.steeringAngleDeg)], [math.radians(CS.steeringRateDeg)]])
    self.x = np.dot(self.A_K, self.x) + np.dot(self.K, y)

    indi_log = log.ControlsState.LateralINDIState.new_message()
    indi_log.steeringAngleDeg = math.degrees(self.x[0])
    indi_log.steeringRateDeg = math.degrees(self.x[1])
    indi_log.steeringAccelDeg = math.degrees(self.x[2])

    steers_des = VM.get_steer_from_curvature(-desired_curvature, CS.vEgo, params.roll)
    steers_des += math.radians(params.angleOffsetDeg)
    indi_log.steeringAngleDesiredDeg = math.degrees(steers_des)

    rate_des = VM.get_steer_from_curvature(-desired_curvature_rate, CS.vEgo, 0)
    indi_log.steeringRateDesiredDeg = math.degrees(rate_des)

    if CS.vEgo < MIN_STEER_SPEED or not active:
      indi_log.active = False
      self.steer_filter.x = 0.0
      output_steer = 0
    else:
      # Expected actuator value
      self.steer_filter.update_alpha(self.RC)
      self.steer_filter.update(last_actuators.steer)

      # Compute acceleration error
      rate_sp = self.outer_loop_gain * (steers_des - self.x[0]) + rate_des
      accel_sp = self.inner_loop_gain * (rate_sp - self.x[1])
      accel_error = accel_sp - self.x[2]

      # Compute change in actuator
      g_inv = 1. / self.G
      delta_u = g_inv * accel_error

      # If steering pressed, only allow wind down
      if CS.steeringPressed and (delta_u * last_actuators.steer > 0):
        delta_u = 0

      output_steer = self.steer_filter.x + delta_u

      output_steer = clip(output_steer, -self.steer_max, self.steer_max)

      indi_log.active = True
      indi_log.rateSetPoint = float(rate_sp)
      indi_log.accelSetPoint = float(accel_sp)
      indi_log.accelError = float(accel_error)
      indi_log.delayedOutput = float(self.steer_filter.x)
      indi_log.delta = float(delta_u)
      indi_log.output = float(output_steer)
      indi_log.saturated = self._check_saturation(self.steer_max - abs(output_steer) < 1e-3, CS)

    return float(output_steer), float(steers_des), indi_log
