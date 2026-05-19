import torch
from torch import nn, optim


class PINN_Solver:
    def __init__(
        self,
        boundaryPoints: torch.Tensor,
        boundaryValues: torch.Tensor,
        collocationPoints: torch.Tensor,
        pdeFn,
        lossFn,
        model: nn.Module,
        initializer,
        optimizer: optim.Optimizer,
    ):

        self.bp = boundaryPoints.detach().clone()
        self.bv = boundaryValues.detach().clone().requires_grad_(False)
        self.cp = collocationPoints.detach().clone()
        self.cv = torch.zeros((len(self.cp), 1), dtype=torch.float64)
        self.pde = pdeFn
        self.loss = lossFn
        self.model = model
        self.initializer = initializer
        self.optimizer = optimizer

        # применени к каждому слою инициализатора
        self.model.apply(PINN_Solver.createMultyLayerinitializer(initializer))

    def trainModel(self, additionalEpochsNum):
        wasTraining = self.model.training

        # переход в режи обучения (есть режим вычисления - eval)
        self.model.train()
        print("start adam training")
        for i in range(additionalEpochsNum):

            # обнуление производных
            self.optimizer.zero_grad()
            loss = self._calculateFullLoss()
            
            # вычисление частных производных по каждому весу
            loss.backward()

            # шаг оптимизатора (град. спуск)
            self.optimizer.step()

        print("end adam training")
        self.model.train(wasTraining)

    # дотренировка модели с помощью LBFGS
    # в статье использовался только этот оптимизатор с maxIter = 50000
    def trainLBFGS(self, maxIter):
        wasTraining = self.model.training

        self.model.train()
        optimizer = optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=maxIter,
            max_eval=maxIter,
            history_size=50,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = self._calculateFullLoss()
            loss.backward()
            return loss

        optimizer.step(closure)
        self.model.train(wasTraining)

    def getSolution(self, points: torch.Tensor):
        return self.model(points)

    def _calculateFullLoss(self):
        bp = self.bp
        # detach для сборса вычисленных производных
        # так как каждое вычисление производных прибавляет результат
        cp = self.cp.detach().requires_grad_(True)

        boundary_pred = self.model(bp)
        boundary_loss = self.loss(boundary_pred, self.bv)

        collocation_pred = self.model(cp)
        collocation_residual = self.pde(cp, collocation_pred)

        # self.cv - нулевой тензор
        collocation_loss = self.loss(collocation_residual, self.cv)

        return boundary_loss + collocation_loss

    @staticmethod
    def createMultyLayerinitializer(initializer):
        def initWeights(m: nn.Module):
            if isinstance(m, nn.Linear):
                initializer(m.weight)
                nn.init.zeros_(m.bias)

        return initWeights
