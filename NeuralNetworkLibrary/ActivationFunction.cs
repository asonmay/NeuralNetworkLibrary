using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    public class ActivationFunction
    {
        Func<double, double> function;
        Func<double, double> derivative;
        public ActivationFunction(Func<double, double> function, Func<double, double> derivative)
        {
            this.function = function;
            this.derivative = derivative;
        }

        public double Function(double input)
        {
            return function(input);
        }

        public double Derivative(double input) 
        {
            return derivative(input);
        }
    }
}
