import copy
import numpy as np
import progressbar as pb

def value_iteration(number_actions, number_states, transitions, rewards, theta=0.0001, discount_factor=1.0):
    def one_step_lookahead(state, V):
        A = np.zeros(number_actions)
        for a in range(number_actions):
            for next_state in range(number_states):
                if transitions[a][state, next_state]:
                    A[a] += transitions[a][state, next_state] * (rewards[a][state, next_state] + discount_factor * V[next_state])
        return A
    
    V = np.zeros(number_states)
    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        bar = pb.ProgressBar()
        for s in bar(range(number_states)):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10. 
            V[s] = best_action_value        
        # Check if we can stop 
        print(delta)
        if delta < theta:
            break
    
    # Create a deterministic policy using the optimal value function
    policy = np.zeros(number_states)
    for s in range(number_states):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s] = best_action
    return policy, V

def valueIteration(states, actions, transitions, rewards, epsilon, discount):
    num_states = len(states)
    values = np.random.uniform(size=num_states)
    delta = 0
    while True:
        bar = pb.ProgressBar()
        for state_index in bar(range(len(states))):
            potential_values  = []
            for action in actions:
                action_result = 0
                for next_state_index in range(len(states)):
                    action_result += transitions[action][state_index, next_state_index] * (rewards[action][state_index, next_state_index] + discount* values[next_state_index])
                potential_values.append(action_result)
            best_action_value = np.max(potential_values)
            delta = max(delta, np.abs(best_action_value - values[state_index]))
            values[state_index] = best_action_value
        print(delta)
        if delta < epsilon:
            break
    policy = np.zeros(num_states) - 1
    bar = pb.ProgressBar()
    for state_index in bar(range(len(states))):
        potential_values  = []
        for action in actions:
            action_result = 0
            for next_state_index in range(len(states)):
                action_result += transitions[action][state_index, next_state_index] * (rewards[action][state_index, next_state_index] + discount* values[next_state_index])
            potential_values.append(action_result)
        policy[state_index] = np.argmax(potential_values)
    return values, policy
