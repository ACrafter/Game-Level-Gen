######### GenV1 Iterations:

"""


"""

######### GenV3 Iterations:

"""
PPO#1:

Reward Function:Initially at zero
                0.1 for each number with range and 0.2 for each prime number with range (-1 for both if out of range)
                once 25 steps are completed if the level is playable PCGRL reward for the lost_lives and time_spent
                else -40

Results: Inconsistency in generating playable levels (too low prime numbers)
"""

"""
PPO#8:

Reward: Initially at -10
        +0.1 for each number with range and +0.2 for each prime with range (-1 for both if out of range)
        once every 25 steps are completed if the level is playable +PCGRL * 2 reward for the lives and time 

Training setup: 
    5 Parallel Environment using SubprocVecEnv for 5,000,000 Steps with player training 1,000,000 every 1,000,000 steps
    
Results: Very early on fixation on generating full sets of prime numbers (To reduce the -10 to -5)
         Training ended at 2.5 Mil Steps (Constant Entropy Loss)
"""

"""
PPO#9:

Reward: Initially 0
        for the first 25 steps:
            if number in range:
                + 0.1
            else:
                - 1
            
            if prime count in range:
                + 0.2
            else:
                - 2
        
        After 25th step
        if level is playable:
            +10 + 2 *  rewards for lives
        else:
            -20
            
Training Setup: Same as PPO#8 but the Generator freezes after 500,000 steps
Results: The model is able to generate playable, diverse levels maintaining the number ranges.
"""
