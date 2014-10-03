using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningClub
{
    class PerceptronTester
    {
        static void Main(string[] args)
        {
            // These test cases suck btw
            double[][] input = new double[][]
            {
                new double[]{1,1,0,1,1,0,1,0,1},
                new double[]{0,0,1,0,1,1,1,1,1},
            };
            double[] output = new double[] { 1, 0 };

            PerceptronNetwork network = new PerceptronNetwork(9);
            PerceptronTeacher teacher = new PerceptronTeacher(network, 0.05D);

            for (int x = 0; x < 1000; x++)
            {
                for (int i = 0; i < 2; i++)
                {
                    double error = teacher.Teach(input[i], output[i]);
                    Console.WriteLine(error);
                }
            }
            
        }
    }

    public class PerceptronNetwork
    {
        public readonly int Inputs;
        private double[] weights;

        public double this[int index]
        {
            get { return this.weights[index]; }
            set { this.weights[index] = value; }
        }

        private Random random = new Random();

        public PerceptronNetwork(int inputs)
        {
            this.Inputs = inputs;
            this.weights = Enumerable.Range(0, this.Inputs)
                .Select(x => random.NextDouble() * 2 - 1)
                .ToArray();
        }

        public double Run(double[] inp)
        {
            Trace.Assert(weights.Length == inp.Length);
            double sum = Enumerable
                .Range(0, weights.Length)
                .Select(x => weights[x] * inp[x])
                .Sum();

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
        public double Alpha { get; set; }

        public PerceptronTeacher(PerceptronNetwork network, double alpha)
        {
            this.Network = network;
            this.Alpha = alpha;
        }

        public double Teach(double[] input, double output)
        {
            double sum = 0;
            double a = output - this.Network.Run(input);

            for (int i = 0; i < this.Network.Inputs; i++)
            {
                this.Network[i] += this.Alpha * a * input[i];
                sum += a;
            }
            return sum;
        }
    }
}
