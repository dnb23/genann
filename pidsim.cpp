#define DEBUG 1

#include "genann.h"
#include <cstdio>

using namespace std;


double tempTarget = 102.5; // Intial Target

double K;
double preK;
double preActivation;
genann_actfun actfuns[3] = {genann_act_sigmoid,custom_act_sigmoid_I,custom_act_sigmoid_D};

genann* ann;
const double learnRate = 0.00005;

const double roomTemp = 21;
const double isolFactor = 0.001;
const double heatPower = 1;
const double heatCapac = 5;
double cooler = 0;

const double dT = 0.5;

double pwm = 0;


double ps[] = {3.39205142362272837886e-01,2.11966368572538491066e-02,3.17326108811751994576e-01,2.99019218672453790386e-01,3.64386058980071370073e-01,-3.63601515108180228797e-01,-3.36447141623702872248e-01,3.99294464750426456590e+01,1.02270882242008962493e+01,1.60430374488675875355e+00,-1.57125167881785565704e+00,-1.58970051594125871830e+00,-1.06639817350998478140e+00};


double getTemp() {
  double diff = (roomTemp - K) * isolFactor * dT + (pwm) * heatPower * heatCapac * dT + heatCapac * cooler * dT;
  //  printf("%f,%f,%f,%f",(roomTemp - K) * isolFactor * dT,dT,diff,pwm);
  return K + diff;
}

int load = 0;

void setup() {
  ann =  genann_init(2, 1, 3, 1,actfuns);

  if (load) {

  for (int i=0;i < ann->total_weights;++i) {
    ann->weight[i] = ps[i];
  }
  }

  K = 21;
  preK = K;
   // Setup Neural Network

   // First run of network
   K = getTemp();
   double inputs[2] = {K,tempTarget};
   genann_run(ann,inputs);
   printf("K,PWM\n");
   // Maybe Load Weights from EEPROM
}

double sign(double x) {
  if (x < 0) {
    return -1;
  } else if (x > 0) {
    return 1;
  } else {
    return 0;
  }
}

void heatTarget(double target) {


  K = getTemp();

  double diffK = K - preK;



  double *activation = ann->output + ann->total_neurons - 1;

  double dadk = 0;

  if ((*activation - preActivation) != 0) {
    dadk = 1; //sign(diffK/(*activation - preActivation));
  }


  double outputs[1] = {target};
  double inputs[2] = {K,target};

  // Train for previous step
  genann_train(ann,inputs,outputs,learnRate,K,dadk);
  preActivation = *(activation);
  genann_run(ann,inputs);
  //8bit activation
  int activation8 = (255 * (*activation));
  pwm = *(activation);
  printf("%f,%i\n",K,activation8);


}

void steamMode() {
  // PIN ON!
  K = getTemp();
}

void loop() {

  //handleInputs();
  int mode = 1;
  //  tempTarget = 90;

  if (mode == 0) {
    // Nothing
  } else if (mode == 1) {
    heatTarget(tempTarget);
  } else if (mode == 2) {
    steamMode();
  }

}


int test() {
  setup();

  for (int i=0;i<50;++i) {
    loop();
  }


  double inputs[2] = {50,102.5};
  genann_run(ann,inputs);


  printf("Deltas:\n");
  for (int i=0;i < ann->total_neurons-ann->inputs;++i) {
    printf("%f,",ann->delta[i]);
  }
  printf("\nOutputs:\n");
  for (int i=0;i < ann->inputs+ann->hidden * ann->hidden_layers + ann->outputs;++i) {
    printf("%f,",ann->output[i]);
  }


  genann_run(ann,inputs);
  ann->weight[9] += 0.01;

    printf("Deltas:\n");
  for (int i=0;i < ann->total_neurons-ann->inputs;++i) {
    printf("%f,",ann->delta[i]);
  }
  printf("\nOutputs:\n");
  for (int i=0;i < ann->inputs+ann->hidden * ann->hidden_layers + ann->outputs;++i) {
    printf("%f,",ann->output[i]);
  }


}

int main() {
  setup();

  FILE *out = fopen("out.txt", "w");

  for (int i=0;i<10000;i++) {
    loop();
  }


  tempTarget = 110;
  for (int i=0;i<10000;i++) {
    loop();
  }


  tempTarget = 105;
  for (int i=0;i<10000;i++) {
    loop();
  }

  tempTarget = 115;
  for (int i=0;i<10000;i++) {
    loop();
  }


  tempTarget = 105;
  for (int i=0;i<10000;i++) {
    loop();
  }


  tempTarget = 20;
  for (int i=0;i<10000;i++) {
    loop();
  }


    tempTarget = 130;
  for (int i=0;i<10000;i++) {
    loop();
  }

  tempTarget = 30;
  for (int i=0;i<10000;i++) {
    loop();
  }


  genann_free(ann);


}
