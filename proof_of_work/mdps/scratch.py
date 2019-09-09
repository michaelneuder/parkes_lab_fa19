    def getRhoBounds(self):
        lowRho = 0
        highRho = 1
        while(highRho - lowRho > self.epsilon/8):
            rho = (highRho + lowRho) / 2
            Wrho = []
            for i in range(self.action_count):
                Wrho.append((1-rho)*self.reward_selfish[i] - rho*self.reward_honest[i])
            rvi = mdptoolbox.mdp.RelativeValueIteration(self.transitions, Wrho, self.epsilon/8)
            rvi.run()
            lowerBoundPolicy = rvi.policy
            reward = rvi.average_reward
            if reward > 0:
                lowRho = rho
            else:
                highRho = rho
        print('alpha: ', self.alpha, 'lower bound reward:', rho)
        lowerBoundRho = rho
        lowRho = rho
        highRho = min(rho+0.1, 1)
        while (highRho - lowRho) > (self.epsilon / 8):
            rho = (highRho + lowRho) / 2
            self.makeRewardsOverpay(rho)
            Wrho = []
            for i in range(self.action_count):
                Wrho.append((1-rho)*self.reward_selfish[i] - rho*self.reward_honest[i])
            rhoPrime = max(lowRho - self.epsilon/4, 0)
            rvi = mdptoolbox.mdp.RelativeValueIteration(self.transitions, Wrho, self.epsilon/8)
            rvi.run()
            reward = rvi.average_reward
            policy = rvi.policy
            if reward > 0:
                lowRho = rho
            else:
                highRho = rho
        print('alpha: ', self.alpha, 'upper bound reward', rho)