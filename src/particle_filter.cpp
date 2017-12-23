/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <unordered_map>

#include "particle_filter.h"

using namespace std;

namespace {
  LandmarkObs transformObservation(const LandmarkObs& ob, double x, double y, double theta) {
    LandmarkObs transformed_ob;
    transformed_ob.x = x + cos(theta) * ob.x - sin(theta) * ob.y;
    transformed_ob.y = y + sin(theta) * ob.x + cos(theta) * ob.y;
    return transformed_ob;
  }

  LandmarkObs findClosestObservationAroundLandmark(const Map::single_landmark_s& landmark,
                                                   const vector<LandmarkObs>& observations) {
    double min_dist = numeric_limits<double>::infinity();
    int ob_index = 0;
    for (auto i = 0; i < observations.size(); ++i) {
      const auto ob = observations[i];
      auto dist = dist(landmark.x_f, landmark.y_f, ob.x, ob.y);
      if (dist < min_dist) {
        min_dist = dist;
        ob_index = i;
      }
    }
    return observations[ob_index];
  }

  double getMaxParticleWeight(const vector<Particle> particles) {
    double max_weight = 0;
    for (auto particle : particles) {
      max_weight = max(max_weight, particle.weight);
    }
    return max_weight;
  }
} // namespace

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 100;

  auto x_std = std[0];
  auto y_std = std[1];
  auto theta_std = std[2];

  random_device rd;
  default_random_engine gen(rd());

  normal_distribution<double> x_dist(x, x_std);
  normal_distribution<double> y_dist(y, y_std);
  normal_distribution<double> theta_dist(theta, theta_std);

  for (auto i = 0; i < num_particles; ++i) {
    Particle p;
    p.id = i;
    p.x = x_dist(gen);
    p.y = y_dist(gen);
    p.theta = theta_dist(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }
}

void ParticleFilter::prediction(double delta_t,
                                double std_pos[],
                                double velocity,
                                double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  auto x_std = std_pos[0];
  auto y_std = std_pos[1];
  auto theta_std = std_pos[2];

  random_device rd;
  default_random_engine gen(rd());

  normal_distribution<double> x_dist(0, x_std);
  normal_distribution<double> y_dist(0, y_std);
  normal_distribution<double> theta_dist(0, theta_std);

  for (auto particle : particles) {
    auto x = particle.x;
    auto y = particle.y;
    auto theta = particle.theta;

    auto x_pred = x + velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
    auto y_pred = y + velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
    auto theta_pred = theta + yaw_rate * delta_t;

    x_pred += x_dist(gen);
    y_pred += y_dist(gen);
    theta_pred += theta_dist(gen);

    particle.x = x_pred;
    particle.y = y_pred;
    particle.theta = theta_pred;
  }
}

void ParticleFilter::updateWeights(double sensor_range,
                                   double std_landmark[],
                                   const std::vector<LandmarkObs>& observations,
                                   const Map& map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation 
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
  const double landmark_std_x = std_landmark[0];
  const double landmark_std_y = std_landmark[1];

  for(auto particle : particles) {
    vector<LandmarkObs> world_obs;
    for (auto i = 0; i < observations.size(); ++i) {
      // Transform each landmark observation from car coordinates into world coordinates (called
      // world observation)
      auto world_ob = transformObservation(observations[i], particle.x, particle.y, particle.theta);
      world_obs.push_back(world_ob);
    }
    for (auto landmark : map_landmarks.landmark_list) {
      auto closest_ob = findClosestObservationAroundLandmark(landmark, world_obs);
      particle.weight *= 1.0 / (2 * M_PI * landmark_std_x * landmark_std_y)
          * exp(-((closest_ob.x - landmark.x_f) * (closest_ob.x - landmark.x_f) / (2.0 * landmark_std_x * landmark_std_x)
                + (closest_ob.y - landmark.y_f) * (closest_ob.y - landmark.y_f) / (2.0 * landmark_std_y * landmark_std_y)));
    }
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  random_device rd;
  default_random_engine gen(rd());

  uniform_real_distribution<double> uniform_dist(0, 1);
  vector<Particle> new_particles;

  // Resampling roulette (wheel) algorithm
  int index = static_cast<int>(uniform_dist(gen) * num_particles);
  double beta = 0.0;
  double max_weight = getMaxParticleWeight(particles);
  for (auto particle : particles) {
    beta += uniform_dist(gen) * 2.0 * max_weight;
    while (beta > particles[index].weight) {
      beta -= particles[index].weight;
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }
  particles.swap(new_particles);
}

Particle ParticleFilter::SetAssociations(Particle& particle,
                                         const std::vector<int>& associations,
                                         const std::vector<double>& sense_x,
                                         const std::vector<double>& sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
