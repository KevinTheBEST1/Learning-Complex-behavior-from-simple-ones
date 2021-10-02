from numpy.random import seed
seed(242)
import tensorflow as tf
import gym
import os
import random

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
import numpy as np
import scipy
import uuid
import shutil

import pandas as pd
import matplotlib.pyplot as plt


import tensorflow.keras.backend as K
import gym_game
import math

from ddpg import OUActionNoise
from ddpg import Buffer

'''
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
'''
env = gym.make('Pygame-v0')

print(f"Input: {env.observation_space}")
print(f"Output: {env.action_space}")

def masked_huber_loss(mask_value, clip_delta):
  def f(y_true, y_pred):
    error = y_true - y_pred
    cond  = K.abs(error) < clip_delta
    mask_true = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    masked_squared_error = 0.5 * K.square(mask_true * (y_true - y_pred))
    linear_loss  = mask_true * (clip_delta * K.abs(error) - 0.5 * (clip_delta ** 2))
    huber_loss = tf.where(cond, masked_squared_error, linear_loss)
    return K.sum(huber_loss) / K.sum(mask_true)
  f.__name__ = 'masked_huber_loss'
  return f


input_shape = (9,) # 8 variables in the environment + the fraction finished we add ourselves
outputs = 4

upper_bound=1
lower_bound=0
# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    #last_init = tf.random_uniform_initializer(minval=0.01, maxval=0.99)

    inputs = layers.Input(shape=(9,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(2, activation="softmax")(out)

    # Our upper bound is 2.0 for Pendulum.
    #outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(9))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(2))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

def create_model(learning_rate, regularization_factor):
  model = Sequential([
    Dense(64, input_shape=input_shape, activation="relu", kernel_regularizer=l2(regularization_factor)),
    Dense(64, activation="relu", kernel_regularizer=l2(regularization_factor)),
    Dense(64, activation="relu", kernel_regularizer=l2(regularization_factor)),
    Dense(outputs, activation='linear', kernel_regularizer=l2(regularization_factor))
  ])
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimizer, loss=masked_huber_loss(0.0, 1.0))
  return model
''' 
def create_model_s(learning_rate, regularization_factor):
  model = Sequential([
    Dense(32, input_shape=input_shape, activation="relu", kernel_regularizer=l2(regularization_factor)),
    Dense(32, activation="relu", kernel_regularizer=l2(regularization_factor)),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(regularization_factor))
  ])
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimizer, loss=masked_huber_loss(0.0, 1.0))
  return model
'''  
def get_q_values(model, state):
  input = state[np.newaxis, ...]
  return model.predict(input)[0]
'''
def get_s_values(model, state):
  input = state[np.newaxis, ...]
  return model.predict(input)[0]
'''
def get_multiple_q_values(model, states):
  return model.predict(states)
'''
def get_multiple_s_values(model, states):
  return model.predict(states)
'''  
def select_action_epsilon_greedy(q_values, epsilon):
  random_value = random.uniform(0, 1)
  random_value=1
  if random_value < epsilon: 
    return random.randint(0, len(q_values) - 1)
  else:
    return np.argmax(q_values)

def select_best_action(q_values):
  return np.argmax(q_values)
  
class StateTransition():

  def __init__(self, old_state, action, reward, new_state, done):
    self.old_state = old_state
    self.action = action
    self.reward = reward
    self.new_state = new_state
    self.done = done

class ReplayBuffer():
  current_index = 0

  def __init__(self, size = 10000):
    self.size = size
    self.transitions = []

  def add(self, transition):
    if len(self.transitions) < self.size: 
      self.transitions.append(transition)
    else:
      self.transitions[self.current_index] = transition
      self.__increment_current_index()

  def length(self):
    return len(self.transitions)

  def get_batch(self, batch_size):
    return random.sample(self.transitions, batch_size)

  def __increment_current_index(self):
    self.current_index += 1
    if self.current_index >= self.size - 1: 
      self.current_index = 0
      
def calculate_target_values(model, target_model, state_transitions, discount_factor):
  states = []
  new_states = []
  for transition in state_transitions:
    states.append(transition.old_state)
    new_states.append(transition.new_state)

  states = np.array(states)
  new_states = np.array(new_states)

  q_values = get_multiple_q_values(model, states)
  q_values_target_model = get_multiple_q_values(target_model, states)

  q_values_new_state = get_multiple_q_values(model, new_states)
  q_values_new_state_target_model = get_multiple_q_values(target_model, new_states)
  
  targets = []
  for index, state_transition in enumerate(state_transitions):
    best_action = select_best_action(q_values_new_state[index])
    best_action_next_state_q_value = q_values_new_state_target_model[index][best_action]
    
    if state_transition.done:
      target_value = state_transition.reward
    else:
      target_value = state_transition.reward + discount_factor * best_action_next_state_q_value

    target_vector = [0, 0, 0, 0]
    target_vector[state_transition.action] = target_value
    targets.append(target_vector)

  return np.array(targets)
'''      
def calculate_target_s_values(model, target_model, state_transitions, discount_factor):
  states = []
  new_states = []
  for transition in state_transitions:
    states.append(transition.old_state)
    new_states.append(transition.new_state)

  states = np.array(states)
  new_states = np.array(new_states)

  #q_values = get_multiple_s_values(model, states)
  #q_values_target_model = get_multiple_s_values(target_model, states)

  #q_values_new_state = get_multiple_s_values(model, new_states)
  #q_values_new_state_target_model = get_multiple_s_values(target_model, new_states)
  
  targets = []
  for index, state_transition in enumerate(state_transitions):
    #best_action = select_best_action(q_values_new_state[index])
    #best_action_next_state_q_value = q_values_new_state_target_model[index][best_action]
    
    if state_transition.done:
      target_value = state_transition.reward
    else:
      target_value = state_transition.reward #+ discount_factor * q_values_new_state_target_model[index][0]
    #0.4*q_values_target_model[index][0] + 0.6
    target_value=sigmoid(target_value)
    #target_vector = [0, 0, 0, 0]
    #target_vector[state_transition.action] = target_value
    targets.append(target_value)
    #print(np.array(targets).shape)

  return np.array(targets)
'''    
def train_model(model, states, targets):
  model.fit(states, targets, epochs=1, batch_size=len(targets), verbose=0)
  
def copy_model(model):
  backup_file = 'backup_'+str(uuid.uuid4())
  model.save(backup_file)
  new_model = load_model(backup_file, custom_objects={ 'masked_huber_loss': masked_huber_loss(0.0, 1.0) })
  shutil.rmtree(backup_file)
  return new_model
  
class AverageRewardTracker():
  current_index = 0

  def __init__(self, num_rewards_for_average=100):
    self.num_rewards_for_average = num_rewards_for_average
    self.last_x_rewards = []

  def add(self, reward):
    if len(self.last_x_rewards) < self.num_rewards_for_average: 
      self.last_x_rewards.append(reward)
    else:
      self.last_x_rewards[self.current_index] = reward
      self.__increment_current_index()

  def __increment_current_index(self):
    self.current_index += 1
    if self.current_index >= self.num_rewards_for_average: 
      self.current_index = 0

  def get_average(self):
    return np.average(self.last_x_rewards)


class FileLogger():

  def __init__(self, file_name='progress.log'):
    self.file_name = file_name
    self.clean_progress_file()

  def log(self, episode, steps, reward, average_reward):
    f = open(self.file_name, 'a+')
    f.write(f"{episode};{steps};{reward};{average_reward}\n")
    f.close()

  def clean_progress_file(self):
    if os.path.exists(self.file_name):
      os.remove(self.file_name)
    f = open(self.file_name, 'a+')
    f.write("episode;steps;reward;average\n")
    f.close()
    
replay_buffer_size = 200000
learning_rate = 0.001
regularization_factor = 0.001
training_batch_size = 128
training_start = 256
max_episodes = 10000
max_steps = 1000
target_network_replace_frequency_steps = 1000
model_backup_frequency_episodes = 100
starting_epsilon = 1.0
minimum_epsilon = 0.01
epsilon_decay_factor_per_episode = 0.995
discount_factor = 0.99
train_every_x_steps = 4

std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(replay_buffer_size, 64)
replay_buffer1 = ReplayBuffer(replay_buffer_size)
replay_buffer2 = ReplayBuffer(replay_buffer_size)

#model1 = create_model(learning_rate, regularization_factor)
#model2 = create_model(learning_rate, regularization_factor)
loss=masked_huber_loss(0.0, 1.0)
model1=tf.keras.models.load_model('/home/kevin/Try1/model_300', custom_objects={loss.__name__: loss})
model2=tf.keras.models.load_model('/home/kevin/Try2/model_300', custom_objects={loss.__name__: loss})
#s1=create_model_s(0.01, regularization_factor)
#s2=create_model_s(0.01, regularization_factor)
target_model1 = copy_model(model1)
target_model2 = copy_model(model2)
#target_model_s1 = copy_model(s1)
#target_model_s2 = copy_model(s2)
epsilon = starting_epsilon
step_count = 0
average_reward_tracker = AverageRewardTracker(100)
file_logger = FileLogger()

for episode in range(max_episodes):
  print(f"Starting episode {episode} with epsilon {epsilon}")

  episode_reward = 0
  state = env.reset()
  fraction_finished = 0.0
  state = np.append(state, fraction_finished)
  
  tf_prev_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
  action = policy(tf_prev_state, ou_noise)
  #print(len(action[0]))
  w1=action[0][0]
  w2=action[0][1]

  first_q_values1 = get_q_values(model1, state)
  #first_q_values1=np.interp(first_q_values1, (first_q_values1.min(), first_q_values1.max()), (-1, +1))
  #2.*(first_q_values1 - np.min(first_q_values1))/np.ptp(first_q_values1)-1
  #np.interp(first_q_values1, (first_q_values1.min(), first_q_values1.max()), (-1, +1))
  first_q_values2 = get_q_values(model2, state)
  #first_q_values2=np.interp(first_q_values2, (first_q_values2.min(), first_q_values2.max()), (-1, +1))
  #2.*(first_q_values2 - np.min(first_q_values2))/np.ptp(first_q_values2)-1
  #np.interp(first_q_values2, (first_q_values2.min(), first_q_values2.max()), (-1, +1))
  #first_s_values1 = get_q_values(s1, state)
  #first_s_values2 = get_q_values(s2, state)  
  first_q_values = (w1)*first_q_values1+(w2)*first_q_values2
  #print(f"Q values1: {first_q_values1}    Q values2: {first_q_values2}")
  #print(f"Max Q: {max(first_q_values)}")

  for step in range(1, max_steps + 1):
    step_count += 1
    
    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(state), 0)   
    action = policy(tf_prev_state, ou_noise)
    w1=action[0][0]
    w2=action[0][1]
    #if(step%50 == 0):
      #print(f"W1= {w1}   W2= {w2}")
    q_values1 = get_q_values(model1, state)
    #print(type(q_values1))
    #q_values1=2.*(q_values1 - np.min(q_values1))/np.ptp(q_values1)-1
    #np.interp(q_values1, (q_values1.min(), q_values1.max()), (-1, +1))
    #print(q_values1)
    q_values2 = get_q_values(model2, state)
    #q_values2= 2.*(q_values2 - np.min(q_values2))/np.ptp(q_values2)-1
    #np.interp(q_values2, (q_values2.min(), q_values2.max()), (-1, +1))
    #s_values1 = get_q_values(s1, state)
    #s_values2 = get_q_values(s2, state) 
       
    action = select_action_epsilon_greedy(w1*q_values1+w2*q_values2, epsilon)
    #action = select_action_epsilon_greedy(q_values1+q_values2, epsilon)
    new_state, reward1,reward2, done, info = env.step(action)
    #print(f"S1: {s_values1} and s2: {s_values2}    r1: {reward1} and r2: {reward2}")
    if(episode>275 and episode<350):
      env.render()
    fraction_finished = (step + 1) / max_steps
    new_state = np.append(new_state, fraction_finished)
    
    episode_reward += (reward1+reward2)

    if step == max_steps:
      print(f"Episode reached the maximum number of steps. {max_steps}")
      done = True

    
    state_transition1 = StateTransition(state, action, reward1, new_state, done)
    state_transition2 = StateTransition(state, action, reward2, new_state, done)
    replay_buffer1.add(state_transition1)
    replay_buffer2.add(state_transition2)
    
    buffer.record((state, [w1,w2], reward1+reward2, new_state))

    state = new_state
    '''
    if step_count % target_network_replace_frequency_steps == 0:
      print("Updating target model")
      target_model1 = copy_model(model1)
      target_model2 = copy_model(model2)
      #target_model_s1=copy_model(s1)
      #target_model_s1=copy_model(s2)
    '''

    if replay_buffer1.length() >= training_start and replay_buffer2.length() >= training_start and step_count % train_every_x_steps == 0:
      '''
      batch1 = replay_buffer1.get_batch(batch_size=training_batch_size)
      batch2 = replay_buffer2.get_batch(batch_size=training_batch_size)
      targets1 = calculate_target_values(model1, target_model1, batch1, discount_factor)
      targets2 = calculate_target_values(model2, target_model2, batch2, discount_factor)
      #ts1=calculate_target_s_values(s1, target_model_s1, batch1, discount_factor)
      #ts2=calculate_target_s_values(s2, target_model_s2, batch2, discount_factor)
      states1 = np.array([state_transition1.old_state for state_transition1 in batch1])
      states2 = np.array([state_transition2.old_state for state_transition2 in batch2])
      train_model(model1, states1, targets1)
      train_model(model2, states2, targets2)
      #train_model(s1, states1, ts1)
      #train_model(s2, states2, ts2)
      '''
      '''
      if(episode>260):
        continue
      else:
      '''
      #print("test")
      buffer.learn(target_actor,target_critic,actor_model,critic_model,gamma,critic_optimizer,actor_optimizer)
      update_target(target_actor.variables, actor_model.variables, tau)
      update_target(target_critic.variables, critic_model.variables, tau)
      

    if done:
      break

  average_reward_tracker.add(episode_reward)
  average = average_reward_tracker.get_average()

  print(f"episode {episode} finished in {step} steps with reward {episode_reward}. Average reward over last 100: {average}")
  file_logger.log(episode, step, episode_reward, average)

  '''
  if episode != 0 and episode % model_backup_frequency_episodes == 0:
    backup_file = f"model_{episode}"
    print(f"Backing up model to {backup_file}")
    model.save(backup_file)
    '''
  epsilon *= epsilon_decay_factor_per_episode
  epsilon = max(minimum_epsilon, epsilon)

data = pd.read_csv(file_logger.file_name, sep=';')

plt.figure(figsize=(20,10))
plt.plot(data['average'])
plt.plot(data['reward'])
plt.title('Reward')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.legend(['Average reward', 'Reward'], loc='upper right')
plt.show()
