/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    /**
     * TODO: Set the number of particles. Initialize all particles to
     *   first position (based on estimates of x, y, theta and their uncertainties
     *   from GPS) and all weights to 1.
     * TODO: Add random Gaussian noise to each particle.
     * NOTE: Consult particle_filter.h for more information about this method
     *   (and others in this file).
     */
    _num_particles = 10;  // TODO: Set the number of particles

    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);

    for(int i=0; i < _num_particles; i++) {
        double sample_x = dist_x(_gen);
        double sample_y = dist_y(_gen);
        double sample_theta = dist_theta(_gen);
        particles.emplace_back(Particle{i, sample_x, sample_y, sample_theta,
                                        1.0, {}, {}, {}});
    }

    _is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
    /**
     * TODO: Add measurements to each particle and add random Gaussian noise.
     * NOTE: When adding noise you may find std::normal_distribution
     *   and std::default_random_engine useful.
     *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
     *  http://www.cplusplus.com/reference/random/default_random_engine/
     */
    for (auto& p : particles) {
        double delta_theta = yaw_rate * delta_t;
        p.x = p.x + velocity / yaw_rate * (sin(p.theta + delta_theta) - sin(p.theta));
        p.y = p.y + velocity / yaw_rate * (cos(p.theta) - cos(p.theta + delta_theta));
        p.theta = p.theta + delta_theta;

        // add noise. put within loop because each particle's mean is different.
        std::normal_distribution<double> dist_x(p.x, std_pos[0]);
        std::normal_distribution<double> dist_y(p.y, std_pos[1]);
        std::normal_distribution<double> dist_theta(p.theta, std_pos[2]);
        p.x = dist_x(_gen);
        p.y = dist_y(_gen);
        p.theta = dist_theta(_gen);
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
    /**
     * TODO: Update the weights of each particle using a mult-variate Gaussian
     *   distribution. You can read more about this distribution here:
     *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
     * NOTE: The observations are given in the VEHICLE'S coordinate system.
     *   Your particles are located according to the MAP'S coordinate system.
     *   You will need to transform between the two systems. Keep in mind that
     *   this transformation requires both rotation AND translation (but no scaling).
     *   The following is a good resource for the theory:
     *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
     *   and the following is a good resource for the actual equation to implement
     *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
     */
    double sum_weight = 0.0;
    for (auto& p : particles) {
        double particle_weight = 1.0;
        for (auto& obs : observations) {
            // 1. convert observation to map coordinate based on particle position and heading.
            // actually observations are ego's (therefore the same for every particle).
            // let observations be each particle's, and if one particle has these observations,
            // how probable that this particle is located at ego's position? (this particle's weight.)
            double obs_x_map = p.x + (cos(p.theta) * obs.x - sin(p.theta) * obs.y);
            double obs_y_map = p.y + (sin(p.theta) * obs.x + cos(p.theta) * obs.y);

            // 2. association. find the closest map landmark from this observation.
            int id = getClosestLandMarkId(obs_x_map, obs_y_map, map_landmarks);
            double mean_x = map_landmarks.landmark_list[id].x_f;
            double mean_y = map_landmarks.landmark_list[id].y_f;

            // 3. calculate weight for this observation.
            double weight = multiv_prob(std_landmark[0], std_landmark[1],
                                        obs_x_map, obs_y_map, mean_x, mean_y);
            particle_weight *= weight;
        }
        p.weight = particle_weight;
        sum_weight += particle_weight;
    }

    // 4. normalize particle weights.
    for(auto& p : particles) {
        p.weight = p.weight / sum_weight;
    }
}

void ParticleFilter::resample() {
    /**
     * TODO: Resample particles with replacement with probability proportional
     *   to their weight.
     * NOTE: You may find std::discrete_distribution helpful here.
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */
    // 1. prepare for discrete distribution generator.
    std::vector<double> weights;
    for (auto& p : particles) {
        weights.push_back(p.weight);
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    // https://en.cppreference.com/w/cpp/numeric/random/discrete_distribution/discrete_distribution
    std::discrete_distribution<> d(weights.begin(), weights.end());

    // 2. generate id for _num_particles times to create new particles.
    std::vector<Particle> new_particles(_num_particles);
    for(int i=0; i<_num_particles; i++) {
        new_particles[i] = particles[d(gen)];
    }
    // replace current particles with new particles.
    particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const std::vector<int>& associations,
                                     const std::vector<double>& sense_x,
                                     const std::vector<double>& sense_y) {
    // particle: the particle to which assign each listed association,
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

std::string ParticleFilter::getAssociations(Particle best) {
    std::vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

std::string ParticleFilter::getSenseCoord(Particle best, std::string coord) {
    std::vector<double> v;

    if (coord == "X") {
        v = best.sense_x;
    } else {
        v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

double ParticleFilter::multiv_prob(double sig_x, double sig_y,
                                   double x_obs, double y_obs,
                                   double mu_x, double mu_y) {
    // calculate normalization term
    double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

    // calculate exponent
    double exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
                      + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));

    // calculate weight using normalization terms and exponent
    double weight = gauss_norm * exp(-exponent);

    return weight;
}

int ParticleFilter::getClosestLandMarkId(double obs_x, double obs_y, const Map &map) {
    int landmark_num = map.landmark_list.size();
    double closest_dist = std::numeric_limits<double>::max();
    int closest_id = -1;
    for(int i=0; i<landmark_num; i++) {
        double x_diff = map.landmark_list[i].x_f - obs_x;
        double y_diff = map.landmark_list[i].y_f - obs_y;
        double current_dist = sqrt(x_diff * x_diff + y_diff * y_diff);
        if (current_dist < closest_dist) {
            closest_dist = current_dist;
            closest_id = i;
        }
    }
    return closest_id;
}
