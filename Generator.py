import random

class Generator:
    """Base class for energy generators"""

    def __init__(self, name, location, capacity, current_production, is_electric=True):
        """
        :param name: Name of the generator
        :param location: Tuple representing the geolocation (latitude, longitude)
        :param capacity: Production capacity (e.g., MW for electric plants)
        :param is_electric: Boolean indicating if the generator produces electricity
        """
        self.name = name
        self.location = location  # (latitude, longitude)
        self.capacity = capacity  # Production capacity in MW or relevant units
        self.is_electric = is_electric  # True if the generator produces electricity
        self.current_production = current_production  # Initialize the current production

    def generate(self, amount):
        """Method to simulate the generator producing energy"""
        if amount <= self.capacity:
            self.current_production = amount
            print(f"{self.name} generates {amount} MW of energy.")
        else:
            self.current_production = self.capacity
            print(f"{self.name} generates only {self.capacity} MW (max capacity).")

    def get_info(self):
        """Method to get information about the generator"""
        return {
            'Name': self.name,
            'Location': self.location,
            'Capacity': self.capacity,
            'Is Electric': self.is_electric,
            'Current Production': self.current_production
        }


# Derived classes for specific types of energy generators
class CoalPlant(Generator):
    """Class for Coal Power Plants"""

    def __init__(self, name, location, capacity, current_production):
        super().__init__(name, location, capacity, current_production, is_electric=True)
        self.fuel_type = "Coal"


class GasPlant(Generator):
    """Class for Gas Power Plants"""

    def __init__(self, name, location, capacity, current_production):
        super().__init__(name, location, capacity, current_production, is_electric=True)
        self.fuel_type = "Gas"


class Renewable(Generator):
    """Class for Renewable Energy Plants (e.g., Solar, Wind, Hydro)"""

    def __init__(self, name, location, capacity, current_production, energy_type):
        """
        :param energy_type: The type of renewable energy (e.g., 'Solar', 'Wind', 'Hydro')
        """
        super().__init__(name, location, capacity, current_production, is_electric=True)
        self.energy_type = energy_type  # Type of renewable energy


class NuclearPlant(Generator):
    """Class for Nuclear Power Plants"""

    def __init__(self, name, location, capacity, current_production, is_on=True):
        """
        :param is_on: Boolean indicating whether the plant is operational (on or off)
        """
        super().__init__(name, location, capacity, current_production, is_electric=True)
        self.is_on = is_on  # Indicates if the plant is currently operational

    def generate(self, amount):
        """Overrides the generate method to account for whether the plant is on or off"""
        if self.is_on ==True and amount >self.capacity:
            self.current_production = self.capacity  # If the plant is on, generate energy as normal
        elif self.is_on == True and 0<amount<self.capacity:
            self.current_production = amount
        
        elif self.is_on == False or amount<0:
            self.current_production = 0  # No production if the plant is off
            print(f"{self.name} is off.")

    def turn_on(self):
        """Turn the nuclear plant on"""
        self.is_on = True
        self.current_production = self.capacity * 0.1  # Simulate some initial production
        print(f"{self.name} is now on.")

    def turn_off(self):
        """Turn the nuclear plant off"""
        self.is_on = False
        self.current_production = 0  # Reset production to 0 when turning off
        print(f"{self.name} is now off.")

    def get_info(self):
        """Method to get information about the nuclear plant"""
        info = super().get_info()
        info['Is On'] = self.is_on
        return info

