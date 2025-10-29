from aero_tools.flow_field import FlowFieldSimulator
from aero_tools.utilities import plot_flow_field

def main():
    pfo = FlowFieldSimulator("./flow_field_input.json")
    plot_flow_field(pfo)

if __name__ == "__main__":
    main()