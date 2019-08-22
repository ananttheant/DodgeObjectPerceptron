using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

[System.Serializable]
public class TrainingSet
{
	public double[] input;
	public double output;
}

/// <summary>
/// Acts as a brain that sends decisions to <see cref="NPC"/>
/// </summary>
public class Perceptron : MonoBehaviour
{
    [TextArea]
    public string Description;

    
    List<TrainingSet> ts = new List<TrainingSet>();
	double[] weights = {0,0};
	double bias = 0;
	double totalError = 0;

#region === Controlling the NPC ===

    /// <summary>
    /// Link to the body here it's NPC
    /// </summary>
    [Header("===============================================")]
    public GameObject NPC;


    /// <summary>
    /// Takes input and desired output
    /// Calculates the input against the weights
    /// And also learns from it by making a training set out of it
    /// </summary>
    /// <param name="i1"> Input 1 </param>
    /// <param name="i2"> Input 2 </param>
    /// <param name="o"> Desired Output </param>
    public void SendInput(double i1, double i2, double o)
    {
        //react
        double result = CalcOutput(i1, i2);
        Debug.Log(result);

        if (result == 0)
        {
            NPC.GetComponent<Animator>().SetTrigger("Crouch");
            NPC.GetComponent<Rigidbody>().isKinematic = false;
        }
        else
        {
            NPC.GetComponent<Rigidbody>().isKinematic = true;
        }

        //Learn from it for the next set
        TrainingSet _set = new TrainingSet();
        _set.input = new double[2] { i1, i2 };

        _set.output = o;
        ts.Add(_set);
        Train();
    }

#endregion

    /// <summary>
    /// Gets the Dot Product of the Current Epoch
    /// </summary>
    /// <param name="v1"></param>
    /// <param name="v2"></param>
    /// <returns></returns>
    double DotProductBias(double[] v1, double[] v2) 
	{
		if (v1 == null || v2 == null)
			return -1;
	 
		if (v1.Length != v2.Length)
			return -1;
	 
		double d = 0;
		for (int x = 0; x < v1.Length; x++)
		{
			d += v1[x] * v2[x];
		}

		d += bias;
	 
		return d;
	}

    /// <summary>
    /// Calculates the output for the current training set against the weights <see cref="weights"/> and the bias <see cref="bias"/>
    /// </summary>
    /// <param name="i"></param>
    /// <returns></returns>
	double CalcOutput(int i)
	{
        //Get the output for the current training set against the weights
		return(ActivationFunction(DotProductBias(weights,ts[i].input)));
	}


    /// <summary>
    /// Used to calculate after the training is done for a custom training set
    /// </summary>
    /// <param name="i1"></param>
    /// <param name="i2"></param>
    /// <returns></returns>
	double CalcOutput(double i1, double i2)
	{
        //Make a training set
		double[] inp = new double[] {i1, i2};

        //Get output
		return(ActivationFunction(DotProductBias(weights,inp)));
	}

    /// <summary>
    /// Main function to decide if the result is true or false
    /// </summary>
    /// <param name="dp"></param>
    /// <returns></returns>
	double ActivationFunction(double dp)
	{
		if(dp > 0) return (1);
		return(0);
	}

    /// <summary>
    /// Initialises the Weights randomly
    /// </summary>
	void InitialiseWeights()
	{
		for(int i = 0; i < weights.Length; i++)
		{
			weights[i] = Random.Range(-1.0f,1.0f);
		}
		bias = Random.Range(-1.0f,1.0f);
	}

    /// <summary>
    /// Updates the Weights based on the Error for the next training set
    /// </summary>
    /// <param name="j"></param>
	void UpdateWeights(int j)
	{
		double error = ts[j].output - CalcOutput(j);
		totalError += Mathf.Abs((float)error);
		for(int i = 0; i < weights.Length; i++)
		{			
			weights[i] = weights[i] + error*ts[j].input[i]; 
		}
		bias += error;
	}

    /// <summary>
    /// Trains the Perceptron
    /// </summary>
	void Train()
	{	
		for(int t = 0; t < ts.Count; t++)
		{
			UpdateWeights(t);
		}
	}

    private void Start()
    {
        InitialiseWeights();
    }

    public void Update()
    {
        if (Input.GetKeyDown("space"))
        {
            InitialiseWeights();
            ts.Clear();
        }
        else if (Input.GetKeyDown("s"))
        {
            SaveWeights();
        }
        else if (Input.GetKeyDown("l"))
        {
            LoadWeights(); 
        }
    }

    #region === Saving and loading Weights ===

    public void SaveWeights()
    {
        string path = Application.dataPath + "/weights.txt";
        var sr = File.CreateText(path);
        sr.WriteLine(weights[0] + "," + weights[1] + "," + bias);
        sr.Close();
    }

    public void LoadWeights()
    {
        string path = Application.dataPath + "/weights.txt";
        if (File.Exists(path))
        {
            var sr = File.OpenText(path);
            string line = sr.ReadLine();
            string[] w = line.Split(',');
            weights[0] = System.Convert.ToDouble(w[0]);
            weights[1] = System.Convert.ToDouble(w[1]);

            bias = System.Convert.ToDouble(w[2]);
            Debug.Log("Loading");
        }
    }

    #endregion

}