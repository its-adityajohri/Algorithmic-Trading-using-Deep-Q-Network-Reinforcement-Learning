from datetime import datetime
import random
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from load_data import fetch_data, plot_prices

class QLearningAgent:

    def __init__(self, possible_actions, input_size):
        self.exploration_rate = 0.95
        self.discount_factor = 0.5
        self.actions = possible_actions
        num_actions = len(possible_actions)

        layer1_size = 200
        layer2_size = 100
        layer3_size = 50

        self.state_input = tf.placeholder(tf.float32, [None, input_size])
        self.q_target = tf.placeholder(tf.float32, [num_actions])

        weights_layer1 = tf.Variable(tf.random_normal([input_size, layer1_size]))
        bias_layer1 = tf.Variable(tf.constant(0.1, shape=[layer1_size]))
        layer1_output = tf.nn.relu(tf.matmul(self.state_input, weights_layer1) + bias_layer1)

        weights_layer2 = tf.Variable(tf.random_normal([layer1_size, layer2_size]))
        bias_layer2 = tf.Variable(tf.constant(0.1, shape=[layer2_size]))
        layer2_output = tf.nn.relu(tf.matmul(layer1_output, weights_layer2) + bias_layer2)

        weights_layer3 = tf.Variable(tf.random_normal([layer2_size, layer3_size]))
        bias_layer3 = tf.Variable(tf.constant(0.1, shape=[layer3_size]))
        layer3_output = tf.nn.relu(tf.matmul(layer2_output, weights_layer3) + bias_layer3)

        weights_output = tf.Variable(tf.random_normal([layer3_size, num_actions]))
        bias_output = tf.Variable(tf.constant(0.1, shape=[num_actions]))
        self.q_values = tf.nn.relu(tf.matmul(layer3_output, weights_output) + bias_output)

        loss = tf.square(self.q_target - self.q_values)
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def choose_action(self, current_state, step_count):
        exploration_threshold = min(self.exploration_rate, step_count / 1000.0)
        if random.random() < exploration_threshold:
            q_values = self.session.run(self.q_values, feed_dict={self.state_input: current_state})
            action_index = np.argmax(q_values)
            selected_action = self.actions[action_index]
        else:
            selected_action = random.choice(self.actions)
        return selected_action

    def update_q_values(self, state, action, reward, next_state):
        current_q_values = self.session.run(self.q_values, feed_dict={self.state_input: state})
        future_q_values = self.session.run(self.q_values, feed_dict={self.state_input: next_state})
        action_index = np.argmax(future_q_values)
        action_idx_in_list = self.actions.index(action)

        current_q_values[0, action_idx_in_list] = reward + self.discount_factor * future_q_values[0, action_index]
        current_q_values = np.squeeze(np.asarray(current_q_values))
        self.session.run(self.optimizer, feed_dict={self.state_input: state, self.q_target: current_q_values})


def simulate_trading(agent, starting_budget, starting_num_assets, price_data, window_size, transaction_fee=0.003, train_agent=True):
    budget = starting_budget
    num_assets = starting_num_assets
    asset_price = 0
    history = []

    for i in range(len(price_data) - window_size - 1):
        if i % 1000 == 0:
            print('Progress {:.2f}%'.format(100 * i / (len(price_data) - window_size - 1)))

        state = np.asmatrix(np.hstack((price_data[i:i + window_size], budget, num_assets)))
        current_portfolio_value = budget + num_assets * asset_price

        action = agent.choose_action(state, i)

        asset_price = float(price_data[i + window_size])
        if action == 'Buy' and budget >= asset_price:
            budget -= (asset_price * (1.0 + transaction_fee))
            num_assets += 1
        elif action == 'Sell' and num_assets > 0:
            budget += (asset_price * (1.0 - transaction_fee))
            num_assets -= 1
        else:
            action = 'Hold'

        new_portfolio_value = budget + num_assets * asset_price

        if train_agent:
            reward = new_portfolio_value - current_portfolio_value
            next_state = np.asmatrix(np.hstack((price_data[i + 1:i + window_size + 1], budget, num_assets)))
            history.append((state, action, reward, next_state))
            agent.update_q_values(state, action, reward, next_state)

    final_portfolio_value = budget + num_assets * asset_price
    return final_portfolio_value


def run_multiple_simulations(agent, initial_budget, initial_assets, price_data, window_size, transaction_fee=0.003):
    total_simulations = 10
    portfolio_results = []

    for _ in range(total_simulations):
        portfolio_value = simulate_trading(agent, initial_budget, initial_assets, price_data, window_size, transaction_fee)
        portfolio_results.append(portfolio_value)
        print('Final portfolio value: ${:.2f}'.format(portfolio_value))

    plt.title('Final Portfolio Values Across Simulations')
    plt.xlabel('Simulation #')
    plt.ylabel('Portfolio Value ($)')
    plt.plot(portfolio_results)
    plt.show()


if __name__ == "__main__":
    price_history = fetch_data("ETH-USD", 3, datetime(2016, 6, 1), datetime(2018, 1, 25), 1, "crypto_prices_1min.npy")

    total_data_points = len(price_history)
    training_data_size = int(total_data_points * 0.90)
    training_prices = price_history[:training_data_size]
    testing_prices = price_history[training_data_size:]

    print("Training Prices:", training_prices)
    print("Testing Prices:", testing_prices)

    plot_prices(training_prices, testing_prices)

    available_actions = ['Buy', 'Sell', 'Hold']
    window_size = 400
    agent = QLearningAgent(available_actions, window_size + 2)
    initial_budget = 100000.0
    initial_assets = 0

    run_multiple_simulations(agent, initial_budget, initial_assets, training_prices, window_size)

    final_portfolio_value = simulate_trading(agent, initial_budget, initial_assets, testing_prices, window_size, train_agent=False)
    print('Final portfolio value after testing: ${:.2f}'.format(final_portfolio_value))
