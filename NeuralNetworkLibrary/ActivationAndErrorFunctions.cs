using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    public static class ActivationAndErrorFunctions
    {
        public static ErrorFunction MSE => new((x, y) => (x - y) * (x - y), (x, y) => 2 * (x - y));

        public static double BinaryStep(double input)
        {
            if (input > 0)
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }

        public static double Identity(double input)
        {
            return input;
        }

        public static double IdentityDerivative(double input)
        {
            return 1;
        }

        public static double TanHDerivative(double input)
        {
            return 1 - Math.Pow(Math.Tanh(input), 2);
        }

        public static double LeakyTanHDerivative(double input)
        {
            double value = TanHDerivative(input);
            return value == 0 ? 0.0000001 : value;
        }

        public static double TanH(double input)
        {
            return Math.Tanh(input);
        }

        public static double BinaryStepDerivative(double input)
        {
            return 0;
        }
    }
}
