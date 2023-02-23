import gymnasium as gym
import numpy as np
import pygame

from bas import wrappers
from bas.render.constants import Colors
from bas.render.renderers.renderer import Renderer
from bas.wrappers.observation import components


class ObservationContainerWrapperRenderer(Renderer):
    """Renderer for drawing all components used in the `ObservationContainerWrapper`."""

    WRAPPER = wrappers.ObservationContainerWrapper

    def __init__(self, render_wrapper: "RenderWrapper", instance: gym.Wrapper):
        super().__init__(render_wrapper, instance)

        self._component_renderers = {
            components.SectionObservationComponent: self.render_section_observation_component,
            components.SectionVelocityObservationComponent: self.render_section_velocity_observation_component,
            components.SectionDistanceObservationComponent: self.render_section_distance_observation_component,
            components.SectionVelocityDistanceObservationComponent: self.render_section_velocity_distance_observation_component,
            components.TargetDirectionObservationComponent: self.render_target_direction_observation_component,
            components.TargetDirectionDistanceObservationComponent: self.render_target_direction_observation_component,
            components.AgentVelocityObservationComponent: self.render_agent_velocity_observation_component,
        }

    def render(self, canvas: pygame.Surface):
        """Distribute rendering of observation components to component renderers."""

        for component, observation in zip(
            self._components, self.cached_step.observation
        ):
            component_renderer = self._component_renderers.get(type(component))
            if component_renderer:
                component_renderer(canvas, component, observation)

    def render_section_observation_component(
        self,
        canvas: pygame.Surface,
        component: components.SectionObservationComponent,
        observation: np.ndarray,
    ):
        agent_to_env_rotation = self._shared_computer.agent_to_env_rotation()
        observation = observation @ agent_to_env_rotation.T

        for section_index, section_position in enumerate(observation):
            self.line(
                canvas,
                Colors.OBSERVATION,
                self.agent.position,
                self.agent.position + section_position * component._max_range,
                2,
            )
            self.circle(
                canvas,
                Colors.OBSERVATION,
                self.agent.position + section_position * component._max_range,
                self.swarm.config.radius / 2,
            )

            angle = (
                section_index * 2 * np.pi / component._num_sections + self.agent.angle + component._offset_angle
            )

            self.line(
                canvas,
                Colors.MISC,
                self.agent.position,
                self.agent.position
                + np.array([np.cos(angle), np.sin(angle)]) * component._max_range,
                1,
            )

        self.circle(canvas, Colors.MISC, self.agent.position, component._max_range, 1)

    def render_section_velocity_observation_component(
        self,
        canvas: pygame.Surface,
        component: components.SectionVelocityObservationComponent,
        section_velocities: np.ndarray,
    ):
        agent_to_env_rotation = self._shared_computer.agent_to_env_rotation()
        section_velocities = section_velocities @ agent_to_env_rotation.T

        section_boid_indices = self._shared_computer.closest_boid_indices_per_section(
            component._num_sections,
            component._max_range + component._subtract_radius * self.env.swarm.config.radius,
        )

        for velocity, boid_index in zip(section_velocities, section_boid_indices):
            if boid_index is None:
                continue

            self.line(
                canvas,
                Colors.OBSERVATION,
                self.swarm.positions[boid_index],
                self.swarm.positions[boid_index]
                + velocity * self.agent.max_velocity * 10,
                2,
            )

        self.circle(canvas, Colors.MISC, self.agent.position, component._max_range, 1)

    def render_section_distance_observation_component(
        self,
        canvas: pygame.Surface,
        component: components.SectionDistanceObservationComponent,
        observation: np.ndarray,
    ):
        for section_index, section_distance in enumerate(observation):
            angle = (
                section_index * 2 * np.pi / component._num_sections + self.agent.angle
            )

            self.line(
                canvas,
                Colors.OBSERVATION,
                self.agent.position,
                self.agent.position
                + np.array(
                    [
                        np.cos(angle + np.pi / component._num_sections),
                        np.sin(angle + np.pi / component._num_sections),
                    ]
                )
                * component._max_range
                * section_distance,
                2,
            )

            self.line(
                canvas,
                Colors.MISC,
                self.agent.position,
                self.agent.position
                + np.array([np.cos(angle), np.sin(angle)]) * component._max_range,
                1,
            )

        self.circle(canvas, Colors.MISC, self.agent.position, component._max_range, 1)

    def render_section_velocity_distance_observation_component(
        self,
        canvas: pygame.Surface,
        component: components.SectionVelocityDistanceObservationComponent,
        observation: np.ndarray,
    ):

        agent_to_env_rotation = self._shared_computer.agent_to_env_rotation()
        section_normals = component._section_normals @ agent_to_env_rotation.T
        section_tangents = component._section_tangents @ agent_to_env_rotation.T

        boid_distances = self._shared_computer.agent_to_boid_distances()
        section_boid_indices = self._shared_computer.closest_boid_indices_per_section(
            component._num_sections,
            component._max_range + component._subtract_radius * self.env.swarm.config.radius,
        )
        section_distances = [
            (
                boid_distances[boid_index]
                - self.env.swarm.config.radius * component._subtract_radius
                if boid_index is not None
                else component._max_range
            )
            for boid_index in section_boid_indices
        ]

        for section_index, (
            section_distance,
            (section_normal_velocity, section_tangent_velocity),
        ) in enumerate(zip(section_distances, observation)):
            angle = (
                section_index * 2 * np.pi / component._num_sections + self.agent.angle
            )

            anchor_position = (
                self.agent.position
                + np.array(
                    [
                        np.cos(angle + np.pi / component._num_sections),
                        np.sin(angle + np.pi / component._num_sections),
                    ]
                )
                * section_distance
            )

            self.line(
                canvas,
                Colors.OBSERVATION,
                anchor_position,
                anchor_position
                + section_normals[section_index]
                * section_normal_velocity
                * self.agent.max_velocity
                * 10,
                4,
            )
            self.line(
                canvas,
                Colors.OBSERVATION,
                anchor_position,
                anchor_position
                + section_tangents[section_index]
                * section_tangent_velocity
                * self.agent.max_velocity
                * 10,
                4,
            )

    def render_target_direction_observation_component(
        self,
        canvas: pygame.Surface,
        component: components.TargetDirectionObservationComponent,
        observation: np.ndarray,
    ):

        agent_to_env_rotation = self._shared_computer.agent_to_env_rotation()
        observation = agent_to_env_rotation @ observation

        self.line(
            canvas,
            Colors.OBSERVATION,
            self.agent.position,
            self.agent.position + observation * 10,
            4,
        )

    def render_agent_velocity_observation_component(
        self,
        canvas: pygame.Surface,
        component: components.AgentVelocityObservationComponent,
        observation: np.ndarray,
    ):
        agent_to_env_rotation = self._shared_computer.agent_to_env_rotation()
        observation = agent_to_env_rotation @ observation
        self.line(
            canvas,
            Colors.OBSERVATION,
            self.agent.position,
            self.agent.position + observation * self.agent.max_velocity * 10,
            8,
        )
