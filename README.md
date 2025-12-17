# REACT
## Using space debris as a distributed sensor for Earth's upper atmosphere.

![210812-nasa-odpo-2x1-mn-0935-a48a1a](https://github.com/user-attachments/assets/04162e51-7b78-40e5-9782-21ceb1b8daef)
> Animation credit: NASA

The REACT framework (Response Estimation and Analysis using Correlated Trajectories) is an innovative approach to space domain awareness and upper-atmosphere science. It repurposes publicly available satellite trajectory data — specifically Two-Line Elements (TLEs) from the U.S. satellite catalog at [Space-Track.org](https://www.space-track.org/auth/login) — to extract actionable insight into the state of satellite operations under varying space weather conditions in low Earth orbit (LEO). Instead of treating each satellite as an independent object, REACT models the collective dynamical response of thousands of satellites, with a particular focus on passive debris objects. This approach reveals how satellites collectively respond to natural forces, such as density variations from geomagnetic storms, and distinguishes propulsive orbital maneuvers by active satellites. While individual TLEs provide only rough, short-term estimates of a satellite’s position, their collective historical record across the entire catalog forms a powerful and underutilized data source. When analyzed in aggregate, these records reveal long-term trends in satellite behavior and the dynamics of Earth’s upper atmosphere that would otherwise remain hidden. 

REACT can utilize correlations across the full tracked catalog of space objects in LEO. It uses “passive” debris populations as a distributed sensor to infer a consensus drag response to space weather, which provides a robust baseline against which individual satellites can be compared. Objects deviating significantly from this baseline can be flagged as potentially maneuvering, improving transparency in satellite operations and enabling more reliable collision-avoidance planning.

<img width="2368" height="1082" alt="image" src="https://github.com/user-attachments/assets/9729d976-934d-4cc4-bacb-e8a75aa01eee" />

# Guide


# Citation
 @article{Parker2026DataDrivenDrag,
  title   = {Data-Driven Satellite Drag Modeling Without Dynamic Knowledge of the Atmosphere},
  author  = {Parker, William E. and Linares, Richard},
  journal = {Journal of Space Weather},
  year    = {2026}
}

# Acknowledgments
Research was sponsored by the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

# Contact
Contact Will Parker via wparker@mit.edu or will@parker42.com. 
