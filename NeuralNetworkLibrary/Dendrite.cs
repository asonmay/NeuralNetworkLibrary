using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    public class Dendrite
    {
        public Neuron Previous { get; }
        public Neuron Next { get; }
        public double Weight { get; set; }
        public double UpdateWeight { get; set; }

        private double previousUpdateWeight;

        public Dendrite(Neuron previous, Neuron next, double weight) 
        {
            Previous = previous;
            Next = next;
            Weight = weight;
        }

        public double Compute()
        {
            return Previous.Output * Weight;
        }

        public void ApplyChanges(double momentum)
        {
            UpdateWeight += previousUpdateWeight * momentum;
            Weight += UpdateWeight;
            previousUpdateWeight = UpdateWeight;
            UpdateWeight = 0;       
        }
    }
}
