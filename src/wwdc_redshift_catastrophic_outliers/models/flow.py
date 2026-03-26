import math 

# Need to make a torch model wrapper for spender
import torch
import torch.nn as nn
import lightning as L
import numpy as np

from torch.func import jvp
from torch.autograd import Function
import torch.nn.functional as F
from torch import Tensor
from wwdc_redshift_catastrophic_outliers.models.modules import get_conditional_len
from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver
from timm.layers import trunc_normal_
from huggingface_hub import PyTorchModelHubMixin

class VelocityField(nn.Module, PyTorchModelHubMixin):
    def __init__(self, code_dim, hidden_dim, conditional_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.vf = nn.Sequential(  # vector field
            nn.Linear(code_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, code_dim),
        )

        self.FiLM_params = nn.Linear(conditional_dim, 2 * hidden_dim)

        self.null_y = nn.Embedding(
            num_embeddings=1,
            embedding_dim=conditional_dim,
        )

    def forward(self, t: Tensor, x_t: Tensor, y: Tensor):
        # flag here if you want conditional and unconditional output.
        if t.ndim == 0:
            t = t.expand(x_t.shape[0])

        null_vector = self.null_y(
            torch.zeros(y.size(0), dtype=torch.long, device=x_t.device)
        )
        x_t = torch.cat([x_t, x_t], dim=0)
        t = torch.cat([t, t], dim=0).unsqueeze(-1)
        x_t = torch.cat([x_t, t], dim=1)

        y = torch.cat([y, null_vector])
        gamma, beta = self.FiLM_params(y).chunk(2, dim=1)
        for idx, layer in enumerate(self.vf):
            x_t = layer(x_t)
            if idx != len(self.vf) - 1 and isinstance(layer, nn.Linear):
                x_t = gamma * x_t + beta
        return x_t

class WrappedModel(nn.Module):
    """Wrapper around velocity model to inject month condition during inference.
    Implements classifier-free guidance according to the formula:
    u ← (1-w)*u_null + w*u_cond
    where:
    - u_null is the velocity with condition dropped
    - u_cond is the velocity with condition intact
    - w is the cfg_scale (default=1.0, which means no guidance)
    """

    def __init__(self, velocity_model):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(self, x, t, **model_extras):
        """Forward pass with classifier-free guidance.

        Args:
            x: Input tensor (batch_size, ...)
            t: Time tensor (batch_size, ) or ()

        Returns:
            Predicted velocity with CFG applied if cfg_scale > 1.0
        """
        cfg_scale = model_extras["cfg_scale"]

        if "r" in model_extras:
            v = self.velocity_model(
                x_t=x, t=t, r=model_extras["r"], y=model_extras["y"]
            )

        batch_size = x.shape[0]
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)

        v = self.velocity_model(x_t=x, t=t, y=model_extras["y"])
        v_cond, v_uncond = torch.chunk(v, chunks=2, dim=0)
        return (1 - cfg_scale) * v_uncond + cfg_scale * v_cond


class LightningFlowMatching(L.LightningModule):
    def __init__(
        self,
        lr,
        batch_size,
        code_dim,
        hidden_dim,
        catalog,
        n_steps=10,
        ckpt_path: str = None,
        method="midpoint"
    ):
        super().__init__()

        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.code_dim = code_dim
        self.lr = lr

        # --- Models --- #
        self.vf = VelocityField(code_dim, hidden_dim, get_conditional_len(catalog))
        self.vf.apply(self._init_weights)

        # --- Load Checkpoints --- #
        if ckpt_path:
            self.vf_state_dict = torch.load(ckpt_path)[
                "state_dict"
            ]  # map_location="cpu"
            self.load_state_dict(self.vf_state_dict, strict=False)
            print("✅ Loaded state dict from checkpoint.")
            self.wrapped_vf = WrappedModel(self.vf)
            # ODE solver hparams
            self.n_steps = n_steps
            self.solver = ODESolver(velocity_model=self.wrapped_vf)
            self.wrapped_vf = WrappedModel(self.vf)
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.method = method
        self.step_size = 1./n_steps

    @property
    def T(self):
        return torch.tensor([1., 0.], device=self.device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        params = list(self.vf.parameters())

        return torch.optim.AdamW(
            params,
            lr=self.lr,
        )


    def base_step(self, batch, partition):
        X, y, _ = batch

        x_0 = torch.randn_like(X)
        t = torch.rand(X.shape[0], device=X.device)

        # sample probability path
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=X)

        # flow matching l2 loss
        ut = self.vf(x_t=path_sample.x_t, y=y, t=path_sample.t)

        u_target = torch.cat([path_sample.dx_t, path_sample.dx_t], dim=0)
        loss = torch.pow(
            ut - u_target,
            2,
        ).mean()
        self.log(f"{partition}_loss", loss)

        return loss

    def training_step(self, batch, _batch_idx):
        return self.base_step(batch, "train")

    def validation_step(self, batch, _batch_idx):
        return self.base_step(batch, "val")

    def test_step(self, batch, _batch_idx):
        return self.base_step(batch, "test")

    def predict_step(self, X, y, embed_opt=["cond"]):
        self.eval()
        with torch.no_grad():
            output = {}

            if "orig" in embed_opt:
                output["orig"] = X
        
            # could reduce this to a single forward pass.
            if "cond" in embed_opt:
                output["cond"] = self.solver.sample(
                    x_init=X,
                    step_size=self.step_size,
                    y=y,
                    cfg_scale=1.0,
                    time_grid=self.T,
                    method=self.method,
                )

            if "uncond" in embed_opt:
                output["uncond"] = self.solver.sample(
                    x_init=X,
                    step_size=self.step_size,
                    y=y,
                    cfg_scale=0.0,
                    time_grid=self.T,
                    method=self.method,
                )
        return output

if __name__ == "__main__":

    def get_conditional_len(catalog):
        return 1  # Mock length

    # 1. Hyperparameters
    CODE_DIM = 32
    HIDDEN_DIM = 128
    BATCH_SIZE = 512
    INPUT_FEATURES = 128
    CATALOG = {} # Mock catalog

    # 2. Setup Data
    # Mock X: (Batch, Features), Mock y: (Batch, Cond_Dim)
    mock_X = torch.randn(BATCH_SIZE, CODE_DIM)
    mock_y = torch.randn(BATCH_SIZE, get_conditional_len(CATALOG))
    batch = (mock_X, mock_y)

    print(get_conditional_len(CATALOG))
    print(mock_X.shape, mock_y.shape)

    # 3. Initialize Model
    model = LightningFlowMatching( # LightningMeanFlowMatching(
        lr=1e-4,
        batch_size=BATCH_SIZE,
        code_dim=CODE_DIM,
        hidden_dim=HIDDEN_DIM,
        catalog=CATALOG
    )

    print("🚀 Initializing test run...")

    # 4. Run a Training Step
    try:
        loss = model.training_step(batch, 0)
        print(f"✅ Success! Training step loss: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()

    # 5. Check VelocityField Output Shape
    # (Testing the forward pass manually)
    z = torch.randn(BATCH_SIZE, CODE_DIM)
    t = torch.rand(BATCH_SIZE)
    y = torch.rand(BATCH_SIZE, 1)

    print(z.shape, t.shape, y.shape)
    vf_out = model.vf(x_t=z, t=t, y=y)
        
    print(f"📊 VelocityField Output Shape: {vf_out.shape} (Expected: [{BATCH_SIZE}, {CODE_DIM}])")
