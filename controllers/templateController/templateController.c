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

#include <math.h>
#include <stdio.h>
#include <webots/compass.h>
#include <webots/gps.h>
#include <webots/keyboard.h>
#include <webots/motor.h>
#include <webots/robot.h>

#define TIME_STEP 16
#define TARGET_POINTS_SIZE 13
#define DISTANCE_TOLERANCE 1.5
#define MAX_SPEED 7.0
#define TURN_COEFFICIENT 4.0

enum XYZAComponents { X = 0, Y, Z, ALPHA };
enum Sides { LEFT, RIGHT };

typedef struct _Vector {
  double u;
  double v;
} Vector;

static WbDeviceTag motors[8];

static bool autopilot = true;

// set left and right motor speed [rad/s]
static void robot_set_speed(double left, double right) {
  int i;
  for (i = 0; i < 4; i++) {
    wb_motor_set_velocity(motors[i + 0], left);
    wb_motor_set_velocity(motors[i + 4], right);
  }
}


// autopilot
static void run_autopilot() {
  double speeds[2] = {0.0, 0.0};
  speeds[LEFT] = MAX_SPEED; 
  speeds[RIGHT] = MAX_SPEED;
  robot_set_speed(speeds[LEFT], speeds[RIGHT]);
}

int main(int argc, char *argv[]) {
  // initialize webots communication
  wb_robot_init();

  wb_robot_step(1000);

  const char *names[8] = {"left motor 1",  "left motor 2",  "left motor 3",  "left motor 4",
                          "right motor 1", "right motor 2", "right motor 3", "right motor 4"};

  // get motor tags
  int i;
  for (i = 0; i < 8; i++) {
    motors[i] = wb_robot_get_device(names[i]);
    wb_motor_set_position(motors[i], INFINITY);
  }


  // start forward motion
  robot_set_speed(MAX_SPEED, MAX_SPEED);

  // main loop
  while (wb_robot_step(TIME_STEP) != -1) {
    if (autopilot)
      run_autopilot();
  }

  wb_robot_cleanup();

  return 0;
}
