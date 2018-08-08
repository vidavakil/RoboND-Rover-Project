import numpy as np

last_rover_steer = 0.0
last_rover_velocity = 0.0
rover_steer_in_a_row = 0.0
picked_up_rock = 0
found_rock_on_right = 0
found_rock_on_left = 0
home_pos = np.zeros(2, dtype=np.float)
first_rock_pos = np.zeros(2, dtype=np.float)
first_rock_pos_step = 0
first_rock_pos_time = 0.0
first_rock_pos_yaw = 0
done_count = 0
backup_time = 0.0
approach_time = 0.0
sim_fps = 13 # The average FPS for which this algorithm was developed.
# Later used this FPS to change waits that were described in steps to seconds.

# This is where you can build a decision tree for determining throttle, brake
# and steer commands based on the output of the perception_step() function
def decision_step(Rover):
    
    global last_rover_steer
    global last_rover_velocity
    global rover_steer_in_a_row
    global picked_up_rock
    global found_rock_on_right
    global found_rock_on_left
    global home_pos
    global first_rock_pos
    global first_rock_pos_time
    global first_rock_pos_step
    global first_rock_pos_yaw
    global done_count
    global backup_time
    global approach_time
    
    # TODO: If we repeatedly get stuck over a small rock, and cannot really get
    # unstuck while over it, then after a couple of tries, we can figure the
    # issue, and after unstucking, go backwards a bit instead of going forward.
    
    # ISSUE: If we get in the sand twice, once going into a narrow place, and
    # once getting out of it, since we tend to turn right on both times, there's
    # a bit of danger that on the way out we may make a U-turn and not stay with
    # the left wall!
    
    # TODO: if we get stuck in front of an obstacle and to our right there is
    # another obstacle so that we cannot turn left, then we will get stuck!
    
    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward': 
            # Check the extent of navigable terrain
            #            if len(Rover.nav_angles) >= Rover.stop_forward:  #!!!!
            old_throttle = Rover.throttle
            
            if len(Rover.adjacent_angles) >= Rover.stop_forward:
                # If mode is forward, navigable terrain looks good
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                
                # Set steering to average angle of adjacent pixels, deviated by
                # a multiple of the std of such angles, plus a random value to
                # help get out of running on a circle, clipped to the range
                # +/- 15
                Rover.steer = np.clip(np.mean(Rover.adjacent_angles * 180/np.pi)
                    + np.std(Rover.adjacent_angles * 180/np.pi) * 0.5, # was 0.4
                    -15, 15) #  + np.random.random_sample() * 1.0

                Rover.message = "Move Forward"
                
                # Adjust the steering angle if it is pointing to an obstacle.
                indices = np.argwhere(np.abs(Rover.nav_angles * 180/np.pi
                    - Rover.steer)  < 3.0)
                mean_distance = np.mean(Rover.nav_dists[indices])
                if mean_distance < 20 or indices.size < 50:
                    Rover.steer -= 5.0
                    Rover.message = "Avoid, Turn"
                
                # Check if too close to an obstacle; if so, go to 'stop' mode.
                indices = np.argwhere(np.abs(Rover.nav_angles * 180/np.pi
                    - Rover.steer)  < 10.0)
                far_indices = np.argwhere(Rover.nav_dists > 10)
                obstacle_indices = np.intersect1d(far_indices, indices)
                print('Three angle num=', indices.size, 'Far num=',
                    far_indices.size, 'Obstacle num=', obstacle_indices.size)
                if Rover.found_rock == 0 and obstacle_indices.size < 50:
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'
                    Rover.message = "Obstacle, Stop"

                # Check if Rover is stuck; if so, go to 'unstuck' mode.
                distance_moved = np.sqrt(
                    (Rover.previous_pos[9, 0] - Rover.previous_pos[0, 0])**2
                    + (Rover.previous_pos[9, 1] - Rover.previous_pos[0, 1])**2)
                if (distance_moved < 1.0 and Rover.vel < 0.01
                    and old_throttle > 0.0):
                    Rover.steer = -5.0
                    Rover.mode = 'unstuck'
                    Rover.throttle = - Rover.throttle_set
                    Rover.backup_steps = 5.0 + np.random.randint(2) - 1.0
                    backup_time = Rover.total_time + Rover.backup_steps / sim_fps
                    Rover.message = "Unstuck, Turn"

                # Picking up a rock is conditioned on not being stuck. That is,
                # we need an else here.
                else:
                    found_on_right = 0
                    found_on_left = 0
                    if ((Rover.steer - Rover.rock_angle * 180/np.pi) < 45 and
                        (Rover.steer - Rover.rock_angle * 180/np.pi) > -10):
                        found_on_right = 1
                    if ((Rover.rock_angle * 180/np.pi - Rover.steer) < 45 and
                        (Rover.rock_angle * 180/np.pi - Rover.steer) > -10):
                        found_on_left = 1
                    if (found_rock_on_right == 0 and found_rock_on_left == 0
                        and Rover.found_rock == 1):
                        if found_on_left:
                            found_rock_on_left = 1
                        elif found_on_right:
                            found_rock_on_right = 1
                    # Check if you are near a Gold rock; if so, approach it.
                    # Once too close to it, go to 'found' mode.
                    if Rover.found_rock == 1 and \
                        (found_on_right == 1 or found_on_left == 1):
                        distance_to_rock = np.sqrt(
                            (Rover.pos[0] - Rover.rock_pos[0])**2 + \
                            (Rover.pos[1] - Rover.rock_pos[1])**2)
                        if ((distance_to_rock < 6.0 and found_rock_on_right) \
                            or (distance_to_rock < 6.0 and found_rock_on_left)):
                            Rover.throttle = 0
                            # Set brake to stored brake value
                            #if found_rock_on_right:
                            Rover.brake = Rover.brake_set
                            #else:
                            #    Rover.brake = Rover.brake_set * 0.5
                            Rover.mode = 'found'
                            Rover.steer = Rover.rock_angle * 180/np.pi
                            Rover.approach_steps = 400
                            approach_time = Rover.total_time + Rover.approach_steps / sim_fps
                            Rover.message = "Found Rock, Stop"

                        elif distance_to_rock < 20.0:
                            Rover.mode = 'forward'
                            Rover.steer = Rover.rock_angle * 180/np.pi
                            Rover.message = "Found Rock, Approach"
                        #else:
                        #   found_rock_on_left = 0
                        #   found_rock_on_right = 0
        
            # If there's a lack of navigable terrain pixels then go to 'stop'
            # mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                if np.abs(Rover.pitch) < 2.0 or np.abs(Rover.pitch) > 358.0:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'
                    Rover.message = "Blocked, Stop"

                else:
                    # When rover gets stuck in quick sand, its pitch can change
                    # to high, and it will point to the sky and won't see any
                    # terain in front of it and will stop! This will prevent it
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = 0  # Release the brake to allow turning?
                    Rover.steer = -5.0
                    Rover.mode = 'forward'
                    Rover.message = "Quick Sand, Turn"

        # Unstuck the Rover.
        elif Rover.mode == 'unstuck':
            if Rover.total_time - backup_time >= 0:
                Rover.mode = 'forward'
                Rover.throttle = Rover.throttle_set
                Rover.brake = 0
                Rover.steer = 0.0
                Rover.message = "Unstucked"
                print('Unstuck backup time difference', Rover.backup_steps)
            else:
                Rover.throttle = 0
                Rover.backup_steps -= 1
                Rover.mode = 'unstuck'
                Rover.brake = 0
                Rover.steer = -15.0
                Rover.message = "Unstucking"

        # Retract after having picked up a Gold rock.
        elif Rover.mode == 'retract':
            if Rover.total_time - backup_time >= 0:
                Rover.mode = 'forward'
                Rover.throttle = Rover.throttle_set
                Rover.brake = 0
                Rover.steer = 15
                Rover.message = "Retracted"
                print('Retract backup time difference', Rover.backup_steps)
            else:
                if np.abs(Rover.vel) < Rover.max_vel:
                    Rover.throttle = -Rover.throttle_set
                else:
                    Rover.throttle = 0
                Rover.backup_steps -= 1
                Rover.mode = 'retract'
                Rover.brake = 0
                Rover.steer = -1.0
                Rover.message = "Retracting"

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.message = "Slow Down"
            
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a
                # path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line
                    # will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever which way to turn
                    Rover.message = "Now Turn"
                
                # If we're stopped but see sufficient navigable terrain in front
                # then go!
                elif len(Rover.adjacent_angles) >= Rover.go_forward: # else
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(
                        np.mean(Rover.adjacent_angles * 180/np.pi)
                        + np.std(Rover.adjacent_angles * 180/np.pi) * 0.5,
                        -15, 15)
                    Rover.mode = 'forward'
                    Rover.message = "Now Forward"
                else:
                    Rover.steer = -5.0
                    Rover.mode = 'unstuck'
                    Rover.throttle = - Rover.throttle_set
                    Rover.backup_steps = 5 + np.random.randint(2) - 1
                    backup_time = Rover.total_time + Rover.backup_steps / sim_fps
                    Rover.message = "Unstuck, Turn"

        # Found a rock. Now approach it and then pick it up.
        elif Rover.mode == 'found':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.5:
                Rover.mode == 'found'
                Rover.throttle = 0
                #if found_rock_on_right:
                Rover.brake = Rover.brake_set
                #else:
                #    Rover.brake = Rover.brake_set * 0.5
                Rover.steer = Rover.rock_angle * 180/np.pi
                Rover.message = "Approach Rock"
            # If we're not moving (vel < 0.5) then do something else
            elif Rover.near_sample != 0:
                Rover.mode == 'found'
                if Rover.picking_up == 0:
                    Rover.send_pickup = True
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
                    Rover.message = "Pick Up"
                else:
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
                    picked_up_rock = 1
                    Rover.message = "Picking Up"

            elif Rover.found_rock and (Rover.total_time - approach_time < 0):
                Rover.mode = 'found'
                Rover.approach_steps -= 1
                Rover.throttle = Rover.throttle_set
                Rover.brake = 0
                Rover.steer = Rover.rock_angle * 180/np.pi
                Rover.message = "Continue Approach"
            else:
                if Rover.found_rock:
                    print('Approach time difference', Rover.approach_steps)
                # Ideally we want to turn away from this right wall and move
                # back to the left wall, but it is not quite working. Perhaps
                # because being too close to the left wall, other conditions
                # make the rover do a U-turn.
                # TODO: retract a bit, and then move forward. Need to make
                # retract work.
                if picked_up_rock and found_rock_on_right:
                    picked_up_rock = 0
                    Rover.send_pickup = False
                    Rover.mode = 'retract'
                    Rover.backup_steps = 50 # was 40 before changing to time
                    backup_time = Rover.total_time + Rover.backup_steps / sim_fps
                    Rover.throttle = -Rover.throttle_set
                    Rover.brake = 0
                    Rover.steer = -10
                    Rover.message = "Back Up From Right Pick Up"
                elif picked_up_rock and found_rock_on_left:
                    picked_up_rock = 0
                    Rover.send_pickup = False
                    Rover.mode = 'forward'
                    Rover.throttle = 0
                    Rover.brake = 0
                    # Don't change steer
                    if Rover.samples_collected == 1:
                        first_rock_pos[:] = Rover.pos[:]
                        first_rock_pos_step = Rover.step
                        first_rock_pos_time = Rover.total_time
                        first_rock_pos_yaw = Rock.yaw
                    Rover.message = "Forward from Left Pick Up"
                else: # Didn'tpick up
                    picked_up_rock = 0
                    Rover.send_pickup = False
                    Rover.mode = 'forward'
                    Rover.throttle = 0
                    Rover.brake = 0
                    if found_rock_on_right:
                        Rover.steer = 15 # Turn away from the right wall to left
                    else: # Probably rock went out of view. Don't change steer.
                        Rover.steer = 0
                    Rover.message = "Forward, Didn't Pick Up"
                found_rock_on_right = 0
                found_rock_on_left = 0

    # Just to make the rover do something
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        Rover.message = "Move!"

    print(Rover.message)

    print('pos=', Rover.pos, 'yaw=', Rover.yaw, 'velocity=', Rover.vel,
        'steer=', Rover.steer)
    # The following is to help get out of a circular move!
    if (rover_steer_in_a_row >= 0
        and np.abs(Rover.steer - 15) < 6.0
        and np.abs(last_rover_steer - Rover.steer) < 6.0
        and np.abs(last_rover_velocity - Rover.vel) < 0.3
        and np.abs(Rover.vel - Rover.max_vel) < 0.4
        and Rover.mode == 'forward'):
        rover_steer_in_a_row += 1
        print('Counting In Circles=', rover_steer_in_a_row)
        if rover_steer_in_a_row == np.int(200 * Rover.fps / sim_fps):
            rover_steer_in_a_row = -np.int(40 * Rover.fps / sim_fps)
            Rover.steer = np.clip(Rover.steer - 10.0, -15, 15)
            Rover.message = "Getting Out Of Circle"
            print('Getting Out Of Circle', rover_steer_in_a_row)
    else:
        if rover_steer_in_a_row > 0:
            print('Counting Out Circles=', rover_steer_in_a_row)
            rover_steer_in_a_row = 0
    if rover_steer_in_a_row < 0:
        if Rover.mode == 'forward' and Rover.message != "Avoid, Turn":
            rover_steer_in_a_row += 1
            Rover.steer = np.clip(Rover.steer - 10.0, -15, 15)
            Rover.message = "Getting Out Of Circle"
            print('Getting Out Of Circle', rover_steer_in_a_row)
        else:
            rover_steer_in_a_row = 0
            Rover.message = "Got Out Of Circle"
            print('Got Out Of Circle', rover_steer_in_a_row)

    last_rover_steer = Rover.steer
    last_rover_velocity = Rover.vel

    if Rover.step == 1:
        home_pos[:] = Rock.pos[:]

    # The following scheme works only if we never made a wrong turn after
    # picking the first left rock!
    if np.sqrt((Rover.pos[0] - first_rock_pos[0])**2 + \
               (Rover.pos[1] - first_rock_pos[1])**2) < 2.0 \
        and Rover.total_time - first_rock_pos_time > 1000.0 / sim_fps:
        # and Rover.step - first_rock_pos_step > 1000:
        adjusted_first_rock_pos_yaw = first_rock_pos_yaw
        adjusted_rover_yaw = Rover.yaw
        min_of_adjusted = min(adjusted_first_rock_pos_yaw, adjusted_rover_yaw)
        adjusted_first_rock_pos_yaw -= min_of_adjusted
        adjusted_rover_yaw -= min_of_adjusted
        # Check that Rover is pointing to the same direction at which it had
        # seen the rock
        if (np.abs(adjusted_first_rock_pos_yaw - adjusted_rover_yaw) < 60 \
            or np.abs(first_rock_pos_yaw - Rover.yaw) > 305):
            Rover.throttle = 0
            Rover.steer = 0
            Rover.brake = Rover.brake_set
            Rover.mode = 'Done'
            Rover.message = "Done!"
            print('Done')
            if (done_count == 100):
                # TODO: Here, instead of waiting 100 steps, keep changing steer
                # in place, until Rover points towards home_pos. Then, run the
                # algorithm again, walking along left walls, until you reach the
                # home location again (within 20 meters of it). Alternatively,
                # from here, keep going along the left wall until you reach home
                # that is go a full round again.
                exit()
            else:
                done_count += 1

    return Rover
