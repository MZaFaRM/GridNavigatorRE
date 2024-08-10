# Lessons

- The faster the environment the faster the training (ie: lesser timestamps needed)
    - First train the agent in this fast paced environment to test and makes sure everything works right.
    - Then if needed to show others slow the environment speed and train it more

- Decrease the action space at first and slowly introduce new changes, including randomness of environment
- 


# Steps

1. Setup environment started training 2,000,000 timesteps *barely any progress*. standard deviation extremely high
2. Reduced action space to two (left, none, right), fixed car and fuel positions, truncation reduced to 80, *high progress*.
3. Slower car, same result but needs more timestamps to train. 
4. Fast car (10) random fuel, *good progress*.
5. Fast car fuel taking doesn't end game, 3 * higher truncation, to give it time to collect 3 fuel, fuel decreased to 80, just enough to get from one fuel to another *good progress* but extremely variable
6. Created a line to see fuel, if line touches fuel, car gets 2 reward, removed fuel position and speed
7. Realize the entire step 6 was a stupid idea. undo it all

(np.float64(158.59672029860317), np.float64(180.10169370907565))
(np.float64(257.4261049070954), np.float64(167.72771456117022))
(np.float64(321.49870933271944), np.float64(177.21467827566312))
(np.float64(348.58823984444143), np.float64(199.87324965918108))