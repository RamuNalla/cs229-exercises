import numpy as np
from GridWorld import GridWorld

# Value iteration function
# gamma - Discount factor
# conv_threshold - threshold for convergence

# Returns - (V, policy) both are dictionaries: V mapping states to values and policy mapping states to actions

def value_iteration(env, gamma = 0.9, conv_threshold = 1e-12):
    
    # Intialize value function to zeros for all environment states
    V = {state: 0.0 for state in env.states}                      # env.states is a list with all valid states the agent can be (r,c)data

    iteration = 0
    
    while True:                                                     # A loop that runs until convergence
        iteration += 1
        delta = 0                                                   # maximum change in the V across all states - used to check if the updates are stabilizing
        
        # For the current state, consider all actions and with all actions the next states, add the values of that next states. Find the maximum of these next states to add to the current reward
        for state in env.states:                                    # Iterate through all states
            if env.is_terminal(state):
                continue                                            # Skip the terminal states as their value is fixed
            
            v_old = V[state]                                        # Store the old value of the current state in the for loop
            action_values = {}                                      # Calculate the value for each possible action 
            
            for action in env.actions:                              # Looping over the actions - [up, down, left, right]

                next_state, reward, done = env.step(state, action)  # Finding the next state for a state and action, reward - immediate reward for this transition as this is just reward of next state

                if env.is_terminal(next_state) or next_state not in V:
                    next_state_value_in_update = 0.0
                else:
                    next_state_value_in_update = V[next_state]
                # If the next state is terminal, V[next_state] = 0, its future value is 0

                action_values[action] = reward + gamma*next_state_value_in_update
            
            # Once all action values are stored in the array, we need to find the maximum of it
            # Update the state value to the maximum action value
            if action_values:
                V[state] = max(action_values.values())
            else:
                V[state] = v_old

            #Update the maximum change
            delta = max(delta, abs(v_old-V[state]))


        print(f"Iteration {iteration}: max_delta = {delta:.10f}")

        if delta < conv_threshold:
            break   
        
        # Value Iteration converged, now extract the optimum policy
        policy = {}

        for state in env.states:
            if env.is_terminal(state):
                policy[state] = 'TERMINAL'

            for action in env.actions:
                next_state, reward, done = env.step(state, action)
                # use the final V[next_state] here

                if env.is_terminal(next_state) or next_state not in V:
                    next_state_value_in_policy = 0.0
                else:
                    next_state_value_in_policy = V[next_state]

                action_values[action] = reward + gamma * next_state_value_in_policy

            optimal_action = max(action_values, key=action_values.get)                   # action_values is a dictionary with action: value, we need the output action as the optimal_action
            policy[state] = optimal_action
        
        return V, policy
    




if __name__ == "__main__":
    # Define a simple grid layout
    grid_layout = [
        ['.', '.', '.', 'G'],
        ['.', '#', '.', 'P'],
        ['.', '.', '.', '.'],
        ['.', '.', '.', '.']
    ]


    env = GridWorld(grid_layout, living_reward=-0.04, goal_reward=1.0, penalty_reward=-1.0, discount_factor=0.99)

    optimal_values, optimal_policy = value_iteration(env, gamma=env.discount_factor, conv_threshold=1e-4)

    print("\n--- Optimal Value Function (V*) ---")
    
   
    value_grid = np.full((env.rows, env.cols), np.nan)             # Print values in a grid format. Use NaN for non-state cells
    
    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) in env.states:
                 value_grid[r, c] = optimal_values[(r, c)]
            
            cell_value = optimal_values.get((r, c), np.nan)         # Get value or NaN if not a state
            cell_char = env.grid[r][c]
            
            if cell_char == 'G':
                print(f"  G({env.goal_reward:.2f}) ", end="")
            elif cell_char == 'P':
                 print(f"  P({env.penalty_reward:.2f}) ", end="")
            elif cell_char == '#':
                 print("  #      ", end="")
            else:
                 print(f" {cell_value:>6.2f}", end="")            # Print value formatted

        print()                                                     # New line for each row




    print("\n--- Optimal Policy (π*) ---")
    # Print policy actions in a grid format
    policy_grid = np.full((env.rows, env.cols), '')
    action_symbols = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→', 'TERMINAL': '•'} # Symbols for actions

    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            if state in env.states:
                action = optimal_policy.get(state, '')                      # Get action or empty string
                policy_grid[r, c] = action_symbols.get(action, '?')         # Get symbol or '?'

            # Print formatted action symbol or grid cell type
            cell_char = env.grid[r][c]
            if cell_char == 'G':
                print("   • ", end="")
            elif cell_char == 'P':
                 print("   • ", end="")
            elif cell_char == '#':
                 print("   # ", end="")
            else:
                 action = optimal_policy.get((r, c), '')
                 print(f"  {action_symbols.get(action, '?')}  ", end="")

        print() # New line for each row











