import gymnasium as gym
import numpy as np

from swarm.bas import Agent, BASEnv, Blueprint, Swarm, wrappers
from swarm.bas.swarm import InstantSpawner, SwarmConfig
from swarm.bas.wrappers.observation import components

# environment for example scenario in which the difference in strategy becomes visible for an agent trained in a swarm with less (separation range 5) vs. more (separation range 15) separation. Both agent perform differently in this same scenario.
def create_env(
    target_radius=3,
    target_reward=3,
    distance_reward_transform=lambda d: -d,
    collision_termination=False,
    collision_reward=0,
    boid_max_velocity=0.6,
    boid_max_acceleration=0.2,
    num_sections=8,
    separation_range=10,
    cohesion_range=20,
    alignment_range=20,
    steering_weights=(4.0, 2.0, 1.0, 0.0),
    world_size=np.array([200, 200]),
):
    blueprint = Blueprint(
        world_size=world_size,
    )
    agent = Agent(
        radius=1,
        max_velocity=1.0,
        max_acceleration=0.3,
        reset_position=np.array([50,50]),
    )
    swarm = Swarm(
        SwarmConfig(
            num_boids=100,
            radius=2,
            max_velocity=boid_max_velocity,
            max_acceleration=boid_max_acceleration,
            separation_range=separation_range,
            cohesion_range=cohesion_range,
            alignment_range=alignment_range,
            steering_weights=steering_weights,
            target_position=agent.position,
            target_range=20,
            obstacle_margin=3,
            need_for_speed=0.2,
        ),
        InstantSpawner(spawn_positions=np.array([[131.44143939,  88.22481935],
       [148.79294457,  17.58255718],
       [136.80572046, 184.12553982],
       [163.89255137,  94.30087293],
       [ 69.13941451,  57.8756028 ],
       [ 59.75656771, 105.69929033],
       [184.45457183, 135.78581417],
       [121.84614643,  78.62786268],
       [ 84.64211663,  86.63263244],
       [183.81837922, 196.01211191],
       [192.86543658,  21.77014493],
       [ 71.56507031, 184.32429431],
       [  6.66517108, 120.38285977],
       [ 46.04498216, 110.4293714 ],
       [123.55056287, 180.24826579],
       [126.5891923 ,   3.33278743],
       [115.93974973, 144.38608683],
       [180.25115704, 123.72111849],
       [ 97.29940352, 191.79584535],
       [  8.67212107,  99.81232955],
       [ 81.93444865,  87.16940836],
       [156.77012138, 110.99213986],
       [ 11.58006049, 123.63975344],
       [  3.12175936, 181.57914663],
       [178.81973725,  35.1412956 ],
       [117.82881997, 127.81242917],
       [ 55.06631614, 165.25432591],
       [  6.2442374 ,  69.66462094],
       [ 13.47458011, 105.57650311],
       [162.49529305,  22.93399643],
       [ 16.02092589, 169.92982502],
       [ 76.2953419 ,  84.87906205],
       [ 32.30200094, 187.34848695],
       [146.50709448,   3.43101957],
       [ 75.56934682, 190.40364109],
       [ 74.03644951,  99.94261492],
       [ 86.07782545,  25.26226122],
       [ 83.68222407, 114.36095695],
       [102.14115236, 189.0642368 ],
       [ 41.77079891, 165.43388477],
       [106.70874311, 115.75488289],
       [ 93.53653315, 127.94385302],
       [178.14030875,  33.47913867],
       [149.32309577,  12.7027363 ],
       [ 55.09992313,  48.39817559],
       [ 27.39108649, 138.3309049 ],
       [110.76604144, 137.54033315],
       [ 96.436717  , 163.82739836],
       [ 95.64346679, 171.22809194],
       [176.11345534, 177.65723635],
       [ 69.79637557,  41.22354503],
       [ 84.72002167,  37.16904073],
       [166.17471448, 125.83697348],
       [192.78859619,  83.80100789],
       [102.04269543,  18.80414187],
       [122.99207833, 187.39511536],
       [ 17.85935783,  96.69563724],
       [ 91.70464449,  15.12835046],
       [102.7461478 ,  95.77026118],
       [116.17788053,  57.74957075],
       [ 49.85534144, 169.96271167],
       [177.86100016, 144.78735301],
       [ 34.62836911,  89.32954093],
       [ 25.39807445, 133.77634069],
       [ 92.57534156, 100.13452701],
       [101.32360904, 104.0441674 ],
       [176.35572451,  97.77248236],
       [ 97.28892586, 192.91605882],
       [ 10.60207898, 142.86373857],
       [ 99.99708819, 164.56241185],
       [ 53.38341043, 118.48177783],
       [142.60083024,  67.41689461],
       [151.79117244,  25.23942374],
       [139.73973038, 138.06272274],
       [ 40.38644709,  54.42493127],
       [125.76934195,  12.33237956],
       [ 91.18693532,  82.26833761],
       [112.34180327, 190.56174907],
       [139.4439433 , 161.01051164],
       [  2.82669259, 102.87831232],
       [146.19934301,   9.5265283 ],
       [132.51221121,  51.47057342],
       [ 44.06777774,  22.37098363],
       [160.30013987,   6.16966019],
       [ 13.9772468 , 108.48448765],
       [151.37713432, 129.56404819],
       [171.70867176,  36.72179385],
       [ 74.93809592, 184.83559628],
       [156.41484227,   8.45761879],
       [187.21291978,  51.42313908],
       [174.10831901,  68.65354938],
       [ 39.98820109, 164.32592026],
       [ 42.32396967,  15.36128393],
       [ 20.50017307,  79.84412485],
       [ 36.18109818,  62.77017812],
       [191.39647019, 123.30011705],
       [138.70800801,  99.98780807],
       [ 68.27699982,  79.95229263],
       [ 34.41390234,  59.61649982],
       [179.08604653,  21.08108636]])),
    )
    env = BASEnv(blueprint, agent, swarm)

    env.target = np.array([150,150])

    env = wrappers.TargetRewardWrapper(
        env,
        position=env.target,
        target_radius=target_radius,
        target_reward=target_reward,
        distance_reward_transform=distance_reward_transform,
    )
    env = wrappers.BoidCollisionWrapper(
        env,
        collision_termination=collision_termination,
        collision_reward=collision_reward,
        add_reward=True,
    )

    env = wrappers.ObservationContainerWrapper(
        env,
        [
            components.SectionDistanceObservationComponent(num_sections, 20),
            components.SectionVelocityDistanceObservationComponent(num_sections, 20),
            components.TargetDirectionDistanceObservationComponent(env.target, world_size[0]),
            components.AgentVelocityObservationComponent(),
        ],
    )

    env = gym.wrappers.TimeLimit(env, 1000)

    env = wrappers.FlattenObservationWrapper(env)

    env = wrappers.AngularAndVelocityActionWrapper(env)

    env = wrappers.TrajectoryWrapper(env)

    return env
