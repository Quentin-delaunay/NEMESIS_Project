class Source:
    """Base class for primary resource sources"""

    def __init__(self, name, location, reserve, flux_out, is_foreign=False):
        """
        :param name: Name of the source
        :param location: Tuple representing the geolocation (latitude, longitude)
        :param reserve: Total quantity of the resource in the reserve
        :param flux_out: Outgoing flux (how much is extracted per time unit)
        :param is_foreign: Boolean indicating if the source is outside the country (True/False)
        """
        self.name = name
        self.location = location  # (latitude, longitude)
        self.reserve = reserve  # Total amount available in reserve
        self.flux_out = flux_out  # Amount extracted per time unit (e.g., barrels/day)
        self.is_foreign = is_foreign  # Is the source outside the country?
    
    def extract(self, amount):
        """Method to extract a specific amount of resource from the reserve"""
        if self.reserve >= amount:
            self.reserve -= amount
            print(f"{amount} units extracted from {self.name}. Remaining reserve: {self.reserve}")
        else:
            print(f"Not enough reserve in {self.name} to extract {amount} units.")
    
    def get_info(self):
        """Method to get information about the source"""
        return {
            'Name': self.name,
            'Location': self.location,
            'Reserve': self.reserve,
            'Flux Out': self.flux_out,
            'Foreign Source': self.is_foreign
        }


# Derived classes for specific types of resources
class OilWell(Source):
    """Class for Oil Well sources"""

    def __init__(self, name, location, reserve, flux_out, is_foreign=False):
        super().__init__(name, location, reserve, flux_out, is_foreign)
        self.resource_type = "Oil"


class GasField(Source):
    """Class for Gas Field sources"""

    def __init__(self, name, location, reserve, flux_out, is_foreign=False):
        super().__init__(name, location, reserve, flux_out, is_foreign)
        self.resource_type = "Gas"


class UraniumMine(Source):
    """Class for Uranium Mine sources"""

    def __init__(self, name, location, reserve, flux_out, is_foreign=False):
        super().__init__(name, location, reserve, flux_out, is_foreign)
        self.resource_type = "Uranium"