/*
 * Copyright 1996-2024 Cyberbotics Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Included libraries
#include <stdlib.h>                  //for fabs
#include <webots/motor.h>            //obtain motor library
#include <webots/position_sensor.h>  //obtain position sensor library
#include <webots/robot.h>            //obtain main library of webots

// Global defines
#define LEFT 0   // Left side
#define RIGHT 1  // right side
#define INCR 10
#define TIME_STEP 128  // [ms] // time step of the simulation
#define SPEED_UNIT 0.00628
#define ENCODER_UNIT 159.23

int speed[2] = {0, 0};

int main() {
  wb_robot_init();
  WbDeviceTag left_motor, right_motor, left_position_sensor, right_position_sensor;

  /* get a handler to the motors and set target position to infinity (speed control). */
  left_motor = wb_robot_get_device("left wheel motor");
  right_motor = wb_robot_get_device("right wheel motor");
  wb_motor_set_position(left_motor, INFINITY);
  wb_motor_set_position(right_motor, INFINITY);
  wb_motor_set_velocity(left_motor, 0.0);
  wb_motor_set_velocity(right_motor, 0.0);



  wb_robot_cleanup();

  return 0;
}
