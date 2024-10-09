class Transmission:
    """Class representing a transmission line between nodes (edges in the graph)"""

    def __init__(self, name, transmission_type, max_capacity, current_flux=0, is_operational=True):
        """
        :param transmission_type: Type of transmission ('electric', 'pipeline', 'road/train')
        :param max_capacity: Maximum capacity of the transmission (e.g., MW for electric, units of fuel for pipelines)
        :param current_flux: Current flow/flux through the transmission line
        :param is_operational: Boolean indicating whether the transmission line is operational (True=on, False=off)
        """
        self.name = name
        self.transmission_type = transmission_type  # Type: 'electric', 'pipeline', 'road/train'
        self.max_capacity = max_capacity  # Maximum capacity in appropriate units
        self.current_flux = current_flux  # Current flow through the transmission
        self.is_operational = is_operational  # Whether the transmission is on/off

    def transmit(self, amount):
        """Method to simulate transmission of flux through the line"""
        if not self.is_operational:
            self.current_flux = 0
            print(f"The {self.transmission_type} transmission is not operational.")
        elif amount <= self.max_capacity:
            self.current_flux = amount
            print(f"Transmitting {amount} units through {self.transmission_type}.")
        else:
            self.current_flux = self.max_capacity
            print(f"Transmitting only {self.max_capacity} units through {self.transmission_type} (max capacity).")

    def turn_off(self):
        """Method to turn off the transmission (e.g., for maintenance or breakdown)"""
        self.is_operational = False
        self.current_flux = 0  # Reset flux when the transmission is off
        print(f"{self.transmission_type.capitalize()} transmission is now off.")

    def turn_on(self):
        """Method to turn on the transmission after maintenance or repair"""
        self.is_operational = True
        print(f"{self.transmission_type.capitalize()} transmission is now operational.")

    def get_info(self):
        """Method to get information about the transmission line"""
        return {
            'Transmission Type': self.transmission_type,
            'Max Capacity': self.max_capacity,
            'Current Flux': self.current_flux,
            'Is Operational': self.is_operational
        }


# Example usage
if __name__ == "__main__":
    # Create an electric transmission line
    electric_line = Transmission('electric', 1000)

    # Create a pipeline transmission line
    pipeline = Transmission('pipeline', 500)

    # Create a road/train transmission line
    road_train = Transmission('road/train', 200)

    # Print their information
    print(electric_line.get_info())
    print(pipeline.get_info())
    print(road_train.get_info())

    # Simulate transmission of energy/materials
    electric_line.transmit(800)
    pipeline.transmit(600)  # Should hit max capacity
    road_train.transmit(150)

    # Turn off and try transmitting again
    electric_line.turn_off()
    electric_line.transmit(500)

    # Turn it back on and try transmitting again
    electric_line.turn_on()
    electric_line.transmit(500)

    # Print updated information after transmission
    print(electric_line.get_info())
    print(pipeline.get_info())
    print(road_train.get_info())
