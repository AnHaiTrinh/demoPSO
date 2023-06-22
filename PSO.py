import numpy as np

class PSO:
    def __init__(self,
            iteration, # Số lượng vòng lặp
            population, # Số lượng vòng lặp
            w, # Hệ số quán tính
            rp, # Hệ số gia tốc (Thành phần nhận thức)
            rg, # Hệ số gia tốc (Thành phần xã hội)
            x_low, x_high, y_low, y_high, # Khoảng giá trị của biến
            function # Hàm cần cực tiểu hóa
    ):
        b_low, b_up = 0, 1
        self.iteration = iteration
        self.population = population
        self.w = w
        self.rp = rp
        self.rg = rg
        self.x_low = x_low
        self.x_high = x_high
        self.y_low = y_low
        self.y_high = y_high
        self.function = function

        self.positions = []
        self.velocities = []
        self.p_best = []
        for i in range(population):
            x = np.array([np.random.uniform(x_low, x_high), np.random.uniform(y_low, y_high)])
            self.positions.append(x)
            self.p_best.append(x)
            v = np.random.uniform(low=b_low-b_up, high=b_up-b_low, size=2)
            self.velocities.append(v)

        self.positions = np.array(self.positions)
        self.velocities = np.array(self.velocities)
        self.p_best = np.array(self.p_best)
        self.g_best = min(self.p_best, key=self.function)

    def solve(self):
        # Update positions and velocities
        randp = np.random.uniform(0, 1, 2)
        randg = np.random.uniform(0, 1, 2)
        self.velocities = self.velocities * self.w + \
                          self.rp * randp * (self.p_best - self.positions) + \
                         self.rg * randg * (self.g_best - self.positions)
        self.positions = self.positions + self.velocities
        self.positions[:, 0] = np.clip(self.positions[:, 0], self.x_low, self.x_high)
        self.positions[:, 1] = np.clip(self.positions[:, 1], self.y_low, self.y_high)
        for i in range(self.population):
            if self.function(self.positions[i]) < self.function(self.p_best[i]):
                self.p_best[i] = self.positions[i]
                if self.function(self.p_best[i]) < self.function(self.g_best):
                    self.g_best = self.p_best[i]
        return self.positions

    def get_solution(self):
        return self.g_best
