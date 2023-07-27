# Abstract class for properties
from abc import ABC, abstractmethod
import numpy as np

class Property(ABC):
    """
    Abstract Class for molecular properties. All properties must
    inherit from this one.
    """

    # This method below MUST be overridden by whatever property is implemented
    @abstractmethod
    def predict(self, mol, **kwargs):
        """ Given an molecule (format depending on implementation, 
            but usually RDKit molecule or SMILES), 
            returns a value for the property (float)

        Args:
            An RDKit Mol object or list of RDKit mol objects.

        Returns:
            list(float): The estimated values for the property
        """
        pass


    # The methods below are general for every property and will
    # be inherited by all properties. They can be overridden if needed.
    def __init__(self, prop_name, 
                       rew_coeff=1.0,
                       rew_class='hard', rew_acc=None,
                       optimize=False,
                       threshold=0.0,
                       threshold_limit=None,
                       threshold_step=None,
                       **kwargs):
        """ Initializes the class variables that must exist in all properties.
            Other variables may be defined in each property particular
            implementation, as needed.

        Args:
            prop_name (str): The name of the property
            rew_coeff (_type_, optional): Defines this property coefficient in the reward function. 
                                          Defaults to 1.0.
            rew_class (str, optional): Type of reward definition, 'hard' or 'soft'. 
                                       Defaults to 'hard'.
            rew_acc (_type_, optional): Must be modified for soft rewards.
                                        Defines the acceptance ratio of properties outside the
                                        threshold range. 
                                        Defaults to None.
            optimize (bool, optional): Whether or not to optimize this property.
                                       Defaults to False.
            threshold (float, optional): Initial property threshold. Defaults to 0.0.
            threshold_limit (_type_, optional): Must be modified in case of optimization. 
                                                Defines the thresold limit. Defaults to None.
            threshold_step (_type_, optional): _description_. Defaults to None.
        """

        # Name the property
        self.prop_name = prop_name
        self.converged = False
        print(f"When using this property, please cite:\n {self.CITATION}\n")
        
        # All properties start with coeff = 1.0, but can be altered
        try:
            self.rew_coeff = float(rew_coeff)
        except TypeError:
            msg = (f"'rew_coeff = {rew_coeff}' is invalid."
                    "It must be a float.")
            self.bomb_input(msg)
        print(f"  Reward Weight = {self.rew_coeff}")
        
        # Reward class. Must be either 'hard' or 'soft' reward.
        if rew_class.lower() not in ['hard','soft']: 
            msg = (f"'rew_class = {rew_class}' is not a valid reward class. "
                   "It must be either 'hard' or 'soft'.")
            self.bomb_input(msg)
            
        self.rew_class = rew_class.lower()
        print(f"  Reward class = {self.rew_class}")

        # Every property needs a threshold, so that rewards can be calculated.
        # We set a default threshold to 0.0, in case a uer does not want
        # to calculate rewards.
        try:
            self.thresh_ini = float(threshold)
        except TypeError:
            msg=f"'threshold = {threshold}' is invalid. It must a float."
            self.bomb_input(msg)
        print(f"  Initial Threshold = {self.thresh_ini}")
        self.threshold = self.thresh_ini

        # Max and min rewards. Set here so that they can be controlled later if needed.
        self.max_reward = 15
        self.min_reward = 1

        # Reward hook
        self.reward_hook = 0.3
        self.allowed_threshold_jumps = True
        
        # If the reward class is 'soft', we need an acceptance ratio (softness)
        # for values that fall outside the specified threshold:
        if self.rew_class == 'soft':
            try:
                self.rew_acc = float(rew_acc)
            except TypeError:
                msg=(f"'rew_acc = {rew_acc}' invalid or missing, "
                      "but is required by soft rewards.")
                self.bomb_input(msg)
            if not 0.0 < self.rew_acc <= 1.0:
                msg=(f"'rew_acc = {rew_acc}' is invalid."
                      " It must be in the 0.0 < rew_acc < 1.0 interval. "
                      " Either change this value or set the reward class to 'hard'.")
                self.bomb_input(msg)
            print(f"  Acceptance = {self.rew_acc}")

        # Whether or not ot uptimize this property
        self.optimize = bool(optimize)
        print("  Optimize: ", self.optimize)

        if self.optimize:
            # For optimization, we also need the threshold limit and step
            try:
                self.thresh_limit = float(threshold_limit)
            except TypeError:
                msg=(f"'thresh_limit = {threshold_limit}' (invalid or missing), "
                      "but is required for optimization.")
                self.bomb_input(msg)
            print(f"    Threshold Limit = {self.thresh_limit}")

            # Threshold step size
            try:
                self.thresh_step = float(threshold_step)
            except TypeError:
                msg=(f"'thresh_step = {threshold_step}' (invalid or missing), "
                      "but is required for optimization.")
                self.bomb_input(msg)

            print(f"    Threshold Step = {self.thresh_step}")

            # Check threshold limits
            # Initial and final thresholds must be different
            if self.thresh_limit == self.thresh_ini:
                self.bomb_input("Initial and final thresholds must be different")

            # Step size must be =! 0
            if self.thresh_step == 0:
                self.bomb_input("Threshold step size must be =! 0")
            elif (self.thresh_limit - self.thresh_ini) / self.thresh_step < 0:
                self.bomb_input(("Threshold step size (thresh_step) must be in the "
                                 "same direction as the threshold limit (thresh_limit)"))
            elif self.thresh_step > 0:
                self.direction = 'increasing'
            else:
                self.direction = 'decreasing'

            # Done loading this property
            # Finished loading this property.
            print((f"{self.prop_name.upper()} loaded as a {self.direction} property\n"
                   f"with inital threshold {self.thresh_ini} and limit {self.thresh_limit}."))
        else:
            # If not optimizing this property, don't count it in the total reward.
            self.rew_coeff = 0.0

    def bomb_input(self, msg):
        """For sending a message and quitting if there's an error in the input"""        
        message = f"\n*** ERROR when loading {self.prop_name}. ***\n{msg}"
        quit(message)

    def reward(self, prop_values, **kwargs):
        """Given a property value, or list of values,
           returns this property rewards list(float).

        Args:
            prop_value (float or list(floats)): The calculated value(s) of the property

        Returns: 
            list(float): This property rewards for each value passed in.
        """

        _prop_values, rewards = [], []
        _prop_values.extend(prop_values)

        sign = np.sign(self.thresh_step if self.optimize else self.threshold)
        unsigned_threshold = sign * self.threshold

        for value in _prop_values:
            rew = self.min_reward
            if (sign * value) >= (unsigned_threshold):
                rew = self.max_reward
            # TO-DO : Implement soft rewards
            rewards.append(rew)
        return rewards

    def check_and_adjust_property_threshold(self, prop_values):
        """ Checks if the property values are within the threshold range.
            If not, adjusts the threshold accordingly.

        Args:
            prop_values (float or list(floats)): The calculated value(s) of the property
        """
        
        if self.optimize:
            prop_values = np.array(prop_values)

            adjusted = False
            self.converged = False
            if (self.direction == 'increasing'):
                above_thr = np.sum(prop_values > self.threshold) / len(prop_values)

                if (above_thr > self.reward_hook) and (self.threshold < self.thresh_limit):
                    # Enough molecules are better than threshold, tighten it up
                    adjusted = True
                    if self.allowed_threshold_jumps:
                        self.threshold = min(self.thresh_limit,
                                             np.percentile(prop_values, 
                                                           100 - self.reward_hook*100, 
                                                           method='higher'))
                    else:
                        self.threshold += self.thresh_step

                elif (above_thr < self.reward_hook) and (self.threshold >= self.thresh_limit):
                    # Threshold is too tight, need to take a step back
                    adjusted = True
                    print(f"{self.prop_name.upper():25s}:  Threshold too tight, taking a step back...")
                    self.threshold = np.percentile(prop_values, 
                                                   100 - self.reward_hook*100, 
                                                   method='higher')

                # Check convergence limit
                if self.threshold >= self.thresh_limit:
                    self.threshold = self.thresh_limit
                    self.converged = True
                    print(f"{self.prop_name.upper():25s}:  Hooray! Threshold converged!")

                        

            elif (self.direction == 'decreasing'):
                below_thr = np.sum(prop_values < self.threshold) / len(prop_values)

                if (below_thr > self.reward_hook) and (self.threshold > self.thresh_limit):
                    # Enough molecules are better than threshold, tighten it up
                    adjusted = True
                    if self.allowed_threshold_jumps:
                        self.threshold = max(self.thresh_limit,
                                             np.percentile(prop_values,
                                                           self.reward_hook*100, 
                                                           method='lower'))
                    else:
                        self.threshold += self.thresh_step

                elif (below_thr < self.reward_hook) and (self.threshold <= self.thresh_limit):
                    # Threshold is too tight, need to take a step back
                    adjusted = True
                    print(f"{self.prop_name.upper():25s}:  Threshold too tight, taking a step back...")
                    self.threshold = np.percentile(prop_values,
                                                   self.reward_hook*100, 
                                                   method='lower')

                # Check convergence
                if self.threshold <= self.thresh_limit:
                    self.threshold = self.thresh_limit
                    self.converged = True
                    print(f"{self.prop_name.upper():25s}:  Hooray! Threshold converged!")
                    
            if adjusted: 
                print(f"{self.prop_name.upper():25s}:  Threshold adjusted to {self.threshold:6.2f}")                

        return
