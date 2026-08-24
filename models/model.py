import torch.nn as nn

from models.layers import PositionsMLP
from models.transformer import TransformerLayer
from utils.data.dataholder import DataHolder
from exploration.cell_types.embeddings import CellTypeEmbedding, DomainEmbedding
import torch


class Model(nn.Module):
    """
    Model class for the neural network architecture.

    Attributes:
        n_layers (int): Number of transformer layers.
        input_dimensions_node_features (int): Input dimensions for node features.
        input_dimensions_diffusion_time (int): Input dimensions for diffusion time.
        output_dimensions_node_features (int): Output dimensions for node features.
        output_dimensions_diffusion_time (int): Output dimensions for diffusion time.
        mlp_in_node_features (nn.Sequential): MLP for processing input node features.
        mlp_in_diffusion_time (nn.Sequential): MLP for processing input diffusion time.
        mlp_in_position (PositionsMLP): MLP for processing input positions.
        transformer_layers (nn.ModuleList): List of TransformerLayer instances.
        mlp_out_node_features (nn.Sequential): MLP for processing output node features.
        mlp_out_pos (PositionsMLP): MLP for processing output positions.

    Methods:
        __init__(input_dims, n_layers: int, hidden_mlp_dims: dict, hidden_dims: dict, output_dims)
        forward(data: DataHolder) -> DataHolder
    """

    def __init__(
        self,
        input_dims,
        n_layers: int,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        output_dims,
        positionMLP_eps: float = 1e-9,
        embedding_cfg=None,
        num_cell_types: int = 0,
        num_domains: int = 0,
        train_node_features: torch.Tensor = None,
        train_cell_type: torch.Tensor = None,
        build_cell_type_mds: bool = False,
    ) -> None:
        """
        Constructor to initialize the Model instance.

        Args:
            input_dims: Input dimensions.
            n_layers (int): Number of transformer layers.
            hidden_mlp_dims (dict): Dimensions for hidden MLP layers.
            hidden_dims (dict): Dimensions for hidden layers.
            output_dims: Output dimensions.

        Returns:
            None
        """
        super().__init__()
        self.n_layers = n_layers
        self.input_dimensions_node_features = input_dims["node_features_dimensions"]
        self.input_dimensions_diffusion_time = input_dims["diffusion_time_dimensions"]
        self.output_dimensions_node_features = output_dims["node_features_dimensions"]
        self.output_dimensions_diffusion_time = output_dims["diffusion_time_dimensions"]
        self.positionMLP_eps = positionMLP_eps

        act_fn_in = nn.ReLU()
        act_fn_out = nn.ReLU()

        # MLP for processing input node features
        self.mlp_in_node_features = nn.Sequential(
            nn.Linear(self.input_dimensions_node_features, hidden_mlp_dims["X"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]),
            act_fn_in,
        )

        # Optional biological conditioning. The original transcriptome
        # embedding is concatenated with either/both auxiliary embeddings,
        # then projected back to dx so the Transformer stack is unchanged.
        self.cell_type_embedding = None
        self.domain_embedding = None
        self.embedding_fusion = None
        self.cell_type_mds_scale = "mean"
        self.cell_type_mds_weighting = "uniform"
        aux_dim = 0
        cell_cfg = _cfg_section(embedding_cfg, "cell_type")
        if bool(_cfg_value(cell_cfg, "enabled", False)):
            if num_cell_types < 1:
                raise ValueError(
                    "Cell-type embedding enabled but no train cell-type "
                    "vocabulary is available."
                )
            if train_node_features is None or train_cell_type is None:
                raise ValueError(
                    "Cell-type embedding requires train transcriptomes and IDs."
                )
            cell_dim = int(_cfg_value(cell_cfg, "dim", 32))
            self.cell_type_embedding = CellTypeEmbedding.from_features(
                train_node_features,
                train_cell_type.long() - 1,
                n=cell_dim,
                num_classes=int(num_cell_types),
                reserve_unk=True,
                freeze=bool(_cfg_value(cell_cfg, "freeze", False)),
                standardize=bool(_cfg_value(cell_cfg, "standardize", True)),
                pca_max_cells=_optional_int(
                    _cfg_value(cell_cfg, "pca_max_cells", 20000)
                ),
                build_mds=bool(build_cell_type_mds),
            )
            self.cell_type_mds_scale = str(
                _cfg_value(cell_cfg, "mds_scale", "mean")
            )
            self.cell_type_mds_weighting = str(
                _cfg_value(cell_cfg, "mds_weighting", "uniform")
            )
            aux_dim += cell_dim

        domain_cfg = _cfg_section(embedding_cfg, "domain")
        if bool(_cfg_value(domain_cfg, "enabled", False)):
            if num_domains < 1:
                raise ValueError(
                    "Domain embedding enabled but no train domain vocabulary "
                    "is available."
                )
            if train_node_features is None:
                raise ValueError(
                    "Domain embedding requires train transcriptomes for PCA."
                )
            domain_dim = int(_cfg_value(domain_cfg, "dim", 32))
            self.domain_embedding = DomainEmbedding.from_features(
                train_node_features,
                n=domain_dim,
                # One extra row for index 0 = UNK / pad.
                num_domains=int(num_domains) + 1,
                pool=str(_cfg_value(domain_cfg, "pool", "mean")),
                variable_attention=bool(
                    _cfg_value(domain_cfg, "variable_attention", False)
                ),
                reserve_unk=True,
                freeze_pca=bool(
                    _cfg_value(domain_cfg, "freeze_pca", True)
                ),
                attn_hidden=_optional_int(
                    _cfg_value(domain_cfg, "attn_hidden", domain_dim)
                ),
                standardize=bool(
                    _cfg_value(domain_cfg, "standardize", True)
                ),
                pca_max_cells=_optional_int(
                    _cfg_value(domain_cfg, "pca_max_cells", 20000)
                ),
            )
            aux_dim += domain_dim

        if aux_dim > 0:
            fused_dim = int(hidden_dims["dx"]) + aux_dim
            fusion_hidden = int(
                _cfg_value(
                    embedding_cfg,
                    "fusion_hidden_dim",
                    hidden_dims["dx"],
                )
            )
            fusion_dropout = float(
                _cfg_value(embedding_cfg, "fusion_dropout", 0.0)
            )
            self.embedding_fusion = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Linear(fused_dim, fusion_hidden),
                nn.ReLU(),
                nn.Dropout(fusion_dropout),
                nn.Linear(fusion_hidden, hidden_dims["dx"]),
                nn.ReLU(),
            )

        # MLP for processing input diffusion time
        self.mlp_in_diffusion_time = nn.Sequential(
            nn.Linear(self.input_dimensions_diffusion_time, hidden_mlp_dims["y"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]),
            act_fn_in,
        )

        # MLP for processing input positions
        self.mlp_in_position = PositionsMLP(hidden_mlp_dims["pos"])

        # List of TransformerLayer instances
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    node_features_dimensions=hidden_dims["dx"],
                    diffusion_time_dimensions=hidden_dims["dy"],
                    delta_dimensions=hidden_dims["dd"],
                    num_heads=hidden_dims["num_heads"],
                    dim_ff_node_features=hidden_dims["dim_ffX"],
                    dim_ff_diffusion_time=hidden_dims["dim_ffy"],
                    last_layer=False,
                )
                for _ in range(n_layers)
            ]
        )

        # MLP for processing output node features
        self.mlp_out_node_features = nn.Sequential(
            nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["output_features_to_pos_dims"]),
        )

        self.mlp_out_pos_norm = nn.Sequential(
            nn.Linear(
                hidden_dims["output_features_to_pos_dims"] + 3, hidden_mlp_dims["X"]
            ),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["X"], 1),
        )

        # MLP for processing output positions
        self.mlp_out_pos = PositionsMLP(hidden_mlp_dims["pos"])

    def forward(self, data: DataHolder) -> DataHolder:
        """
        Forward pass of the neural network.

        Args:
            data (DataHolder): Input data.

        Returns:
            DataHolder: Output data.
        """
        node_mask = data.node_mask
        node_features = data.node_features
        diffusion_time = data.diffusion_time
        positions = data.positions

        add_diffusion_time_to_out = diffusion_time[
            ..., : self.output_dimensions_diffusion_time
        ]

        # Process input features using MLPs
        transcriptome_embedding = self.mlp_in_node_features(node_features)
        embeddings = [transcriptome_embedding]
        if self.cell_type_embedding is not None:
            if data.cell_type is None:
                raise ValueError(
                    "Cell-type embedding enabled but data.cell_type is missing."
                )
            embeddings.append(self.cell_type_embedding(data.cell_type))
        if self.domain_embedding is not None:
            if data.domain_id is None:
                raise ValueError(
                    "Domain embedding enabled but data.domain_id is missing."
                )
            embeddings.append(
                self.domain_embedding(
                    node_features,
                    data.domain_id,
                    node_mask,
                )
            )
        if self.embedding_fusion is not None:
            node_embedding = self.embedding_fusion(
                torch.cat(embeddings, dim=-1)
            )
        else:
            node_embedding = transcriptome_embedding

        transformed_features = DataHolder(
            node_features=node_embedding,
            diffusion_time=self.mlp_in_diffusion_time(diffusion_time),
            positions=self.mlp_in_position(positions, node_mask),
            node_mask=node_mask,
        ).mask()

        # Apply transformer layers
        for layer in self.transformer_layers:
            transformed_features = layer(transformed_features)

        # Process output features using MLPs
        transformed_node_features = self.mlp_out_node_features(
            transformed_features.node_features
        )

        pos = transformed_features.positions
        norm = torch.norm(pos, dim=-1, keepdim=True)  # bs, n, 1
        new_norm = self.mlp_out_pos_norm(
            torch.cat([transformed_node_features, pos, norm], dim=-1)
        )  # bs, n, 1
        new_pos = pos * new_norm / (norm + self.positionMLP_eps)

        new_pos = new_pos * node_mask.unsqueeze(-1)
        new_pos = new_pos - torch.mean(new_pos, dim=1, keepdim=True)
        pos = new_pos

        # Add input features to output
        transformed_node_features = transformed_node_features
        diffusion_time = add_diffusion_time_to_out

        # Create output DataHolder
        out = DataHolder(
            node_features=transformed_node_features,
            diffusion_time=diffusion_time,
            positions=pos,
            node_mask=node_mask,
        ).mask()

        return out

    def cell_type_mds_loss(self) -> torch.Tensor:
        """Raw transcriptomic-geometry regularizer for the cell-type table."""
        if self.cell_type_embedding is None:
            return next(self.parameters()).new_zeros(())
        return self.cell_type_embedding.mds_loss(
            scale=self.cell_type_mds_scale,
            weighting=self.cell_type_mds_weighting,
        )


def _cfg_section(cfg, name):
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg.get(name)
    return getattr(cfg, name, None)


def _cfg_value(cfg, name, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        value = cfg.get(name, default)
    else:
        value = getattr(cfg, name, default)
    return default if value is None else value


def _optional_int(value):
    return None if value is None else int(value)
