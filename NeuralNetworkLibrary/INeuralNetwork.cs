using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    public interface INeuralNetwork
    {
        public double Fitness { get; set; }
        public NeuralNetwork Network { get; set; }

        public void Reset();
    }
}
