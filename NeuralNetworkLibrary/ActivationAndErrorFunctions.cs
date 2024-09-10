using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    public static class ActivationAndErrorFunctions
    {
        public static double ErrorFunc(double input, double desiredOutput)
        {
            return Math.Pow(input - desiredOutput, 2);
        }

        public static double MeanSquaredErrorDerivative(double output, double desiredOutput)
        {
            return -2 * (desiredOutput - output);
        }

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

        public static double RawValue(double input)
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

        public static double TanH(double input)
        {
            return Math.Tanh(input);
        }
    }
}
