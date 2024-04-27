from djitellopy import Tello

tello = Tello()

tello.connect()
tello.takeoff()

tello.move_left(20)
tello.rotate_counter_clockwise(180)
tello.rotate_counter_clockwise(180)
tello.move_forward(20)

tello.land()
