import torch
import torch.nn as nn
from utils.utils import check_array, init


class Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations
    """
    def __init__(self, args, obs_dim, action_dim, device=torch.device("cpu")):
        super(Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.base = MLPBase(args, obs_dim)
        self.act = ActionDistributionLayer(self.hidden_size, action_dim)

    def forward(self, obs, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        Parameters:
            obs: (np.ndarray / torch.Tensor) observation inputs into network
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode
        
        Returns:
            actions: (torch.Tensor) actions to take
            action_log_probs: (torch.Tensor) log probabilities of taken actions
        """
        obs = check_array(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check_array(available_actions).to(**self.tpdv)
        
        actor_features = self.base(obs)  # the output of base NN
        actions, action_log_probs, action_probs = self.act(actor_features, available_actions, deterministic)  # forward pass
        return actions, action_log_probs, action_probs
    
    def evaluate_actions(self, obs, action, available_actions=None):
        """
        Compute log probability and entropy of given actions
        Parameters:
            obs (np.ndarray or torch.Tensor): Observation inputs into the network.
            action (np.ndarray or torch.Tensor): Actions to evaluate.
            available_actions (np.ndarray or torch.Tensor, optional): Denotes which actions are available to the agent.
                                                                      If None, all actions are available.

        Returns:
            action_log_probs (torch.Tensor): Log probabilities of the given actions.
            dist_entropy (torch.Tensor): Entropy of the action distribution.
        """
        obs = check_array(obs).to(**self.tpdv)
        action = check_array(action).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check_array(available_actions).to(**self.tpdv)
        
        actor_features = self.base(obs)  # the output of base NN
        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action,
                                                                   available_actions)
        return action_log_probs, dist_entropy


class Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
    local observations (IPPO)
    """
    def __init__(self, args, cent_obs_dim, device=torch.device("cpu")):
        super(Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        
        # Initialize the base network (MLPBase) with given arguments and observation dimension
        self.base = MLPBase(args, cent_obs_dim)
        
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        
        # Init the output layer
        self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.to(device)
    
    def forward(self, cent_obs) -> torch.Tensor:
        """
        Compute value function predictions from the given inputs.

        Args:
        - cent_obs (torch.Tensor): Centralized input observations.

        Returns:
        - values (torch.Tensor): Predicted value function outputs.
        """
        cent_obs = check_array(cent_obs).to(**self.tpdv)

        # Pass observations through the base network to extract features
        critic_features = self.base(cent_obs)
        
        # Get value function predictions
        values = self.v_out(critic_features)
        return values



class FixedCategorical(torch.distributions.Categorical):
    """
    Custom categorical distribution that modifies the behavior of sampling and log probabilities.

    Methods:
    - sample(): Samples from the distribution and adds an extra dimension.
    - log_probs(actions): Computes the log probabilities of the given actions.
    - mode(): Returns the mode of the distribution.
    """
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class ActionDistributionLayer(nn.Module):
    """
    Action distribution layer that outputs actions and their log probabilities.

    Args:
    - inputs_dim (int): Dimension of input features.
    - action_dim (int): Dimension of the action space.
    - use_orthogonal (bool): Whether to use orthogonal initialization.
    - gain (float): Gain value for the initialization.

    Methods:
    - forward(x, available_actions=None, deterministic=False): Computes actions and their log probabilities.
    """
    def __init__(self, inputs_dim, action_dim, use_orthogonal=True, gain=0.01):
        super(ActionDistributionLayer, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(inputs_dim, action_dim))

    def forward(self, x, available_actions=None, deterministic=False):
        """
        Computes actions and their log probabilities.

        Args:
        - x (torch.Tensor): Input features.
        - available_actions (torch.Tensor or None): Mask of available actions.
        - deterministic (bool): Whether to use deterministic actions (mode) or sample.

        Returns:
        - actions (torch.Tensor): Sampled actions.
        - action_log_probs (torch.Tensor): Log probabilities of the actions.
        """
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        action_logits = FixedCategorical(logits=x)
        actions = action_logits.mode() if deterministic else action_logits.sample()
        action_log_probs = action_logits.log_probs(actions)
        action_probs = action_logits.probs
        return actions, action_log_probs, action_probs
    
    def evaluate_actions(self, x, action, available_actions=None):
        """
        Compute log probability and entropy of given actions.
        """
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        action_logits = FixedCategorical(logits=x)
        action_log_probs = action_logits.log_probs(action)
        dist_entropy = action_logits.entropy().mean()
        return action_log_probs, dist_entropy


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = nn.ModuleList([nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size)) for i in range(self._layer_N)])

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, obs_dim):
        super(MLPBase, self).__init__()
        self._use_feature_normalization = args.use_feature_normalization  # True
        self._use_orthogonal = args.use_orthogonal  # True
        self._use_ReLU = args.use_ReLU  # True
        self.hidden_size = args.hidden_size
        self._layer_N = args.layer_N  # 2

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)
        
    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x



