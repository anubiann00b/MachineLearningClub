using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// COME TO MACHINE LEARNING CLUB IN O'BYRNE/THOMPSON ROOM ON
// TUESDAYS AND THURSDAYS.
// http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data

namespace MachineLearningClub
{
    /**
    * Simple example of a basic perceptron.
    */
    class PerceptronTester
    {
        static void Main(string[] args)
        {
            double[][] input = new double[569][];
            for (int x = 0; x < input.Length; x++)
            {
                input[x] = new double[32];
            }

            // The expected output for each test case.
            double[] output = new double[569];
            var data = File.ReadAllLines("TextFile1.txt")
                .Select(x =>
                {
                    var left = x.Split(',').Skip(2).Select(y => double.Parse(y)).ToArray();
                    var right = x.Split(',').Skip(1).First().First() == 'M' ? 0 : 1;
                    return new { Data = left, Class = right };
                })
                .ToArray();

            PerceptronNetwork network = new PerceptronNetwork(data.First().Data.Length);
            PerceptronTeacher teacher = new PerceptronTeacher(network, 0.0000005D);

            // Run the training 1000 times, since we don't have enough test cases.
            for (int x = 0; x < 1000; x++)
            {
                for (int i = 0; i < data.Length; i++)
                {
                    double error = teacher.Teach(data[i].Data, data[i].Class);
                }
                var d = data.Select(xx =>
                {
                    var outP = network.Run(xx.Data);
                    if (outP == xx.Class)
                    {
                        return 1;
                    }
                    return 0;
                }).Sum();
                Console.WriteLine((double)d / (double)data.Length);
            }
        }
    }

    public class PerceptronNetwork
    {
        // The number of inputs (dimension).
        public readonly int Inputs;

        // The weight for each input, from -1 to 1.
        private double[] weights;

        // Allows us to do C# magic, like PerceptronNetwork[i].
        public double this[int index]
        {
            get { return this.weights[index]; }
            set { this.weights[index] = value; }
        }

        private Random random = new Random();

        public PerceptronNetwork(int inputs)
        {
            this.Inputs = inputs;

            // Get an array of random values from -1 to 1.
            this.weights = Enumerable.Range(0, this.Inputs)
                .Select(x => random.NextDouble() * 2 - 1)
                .ToArray();
        }

        public double Run(double[] inp)
        {
            // Make sure the user isn't an idiot.
            Trace.Assert(weights.Length == inp.Length);

            // Get the sum of each input multiplied by its corresponding weight.
            double sum = Enumerable
                .Range(0, weights.Length)
                .Select(x => weights[x] * inp[x])
                .Sum();

            // Simple hedge (aka step) function.
            if (sum > 0)
            {
                return 1;
            }
            return 0;
        }
    }

    public class PerceptronTeacher
    {
        public PerceptronNetwork Network { get; private set; }

        // Constant representing training speed.
        public double Alpha { get; set; }

        public PerceptronTeacher(PerceptronNetwork network, double alpha)
        {
            this.Network = network;
            this.Alpha = alpha;
        }

        public double Teach(double[] input, double output)
        {
            // Keep track of the total error. Should be converging to a small value.
            double sum = 0;

            // Error = Expected - Actual
            double error = output - this.Network.Run(input);

            // Based on the error set new weights.
            for (int i = 0; i < this.Network.Inputs; i++)
            {
                // Formula: new weight = old weight * alpha * input value
                this.Network[i] += this.Alpha * error * input[i];
                sum += error;
            }
            return sum;
        }
    }
}