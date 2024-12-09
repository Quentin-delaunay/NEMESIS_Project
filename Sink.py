class Sink:
    """Class representing a consumption point (sink) in the energy system with differentiated demand for resources"""
    
    def __init__(self, name, location, demand=None, is_external=False):
        """
        :param name: Name of the sink (e.g., 'City', 'Plant', etc.)
        :param location: Tuple representing the geolocation (latitude, longitude)
        :param demand: Dictionary to represent the demand for different resources (e.g., {'gas': 0, 'fuel': 0, 'electricity': 0})
        :param is_external: Boolean indicating if the sink is outside the country (external transmission)
        """
        self.name = name
        self.location = location  # (latitude, longitude)
        self.demand = demand if demand else {'gas': 0, 'fuel': 0, 'electricity': 0}  # Initial demand for gas, fuel, electricity
        self.is_external = is_external  # True if the sink is outside the country

    def consume(self, resource, amount):
        """Simulate consumption of a specific resource (gas, fuel, or electricity) by the sink"""
        if resource in self.demand:
            self.demand[resource] += amount
            print(f"{self.name} consumes {amount} units of {resource}.")
        else:
            print(f"Resource type {resource} is not recognized in {self.name}.")

    def get_info(self):
        """Method to get information about the sink"""
        return {
            'Name': self.name,
            'Location': self.location,
            'Demand': self.demand,
            'Is External': self.is_external
        }

