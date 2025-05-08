import numpy as np
from keras import models, layers

class RLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()

    def _build_model(self):
        """Архитектура нейросети"""
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

        def update_model(self, states, actions, rewards):
            """Обновление модели с учетом награды"""
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)

            # Нормализация
            states = (states - np.mean(states, axis=0)) / (np.std(states, axis=0) + 1e-8)
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

            # Обучение
            self.model.fit(states, actions, sample_weight=rewards, epochs=10, verbose=0)
