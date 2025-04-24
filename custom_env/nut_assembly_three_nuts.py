from robosuite.models.objects import RoundNutObject, SquareNutObject
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.models.tasks import ManipulationTask
from robosuite.models.arenas import PegsArena
from robosuite.environments.manipulation.nut_assembly import NutAssembly


class NutAssemblyThreeNuts(NutAssembly):
    def _load_model(self):
        """
        Loads a model with 3 nuts: 1 square, 2 round
        """
        super(NutAssembly, self)._load_model()

        self.nuts = []

        # Define nut objects
        square_nut = SquareNutObject(name="SquareNut")
        round_nut_1 = RoundNutObject(name="RoundNut1")
        round_nut_2 = RoundNutObject(name="RoundNut2")

        self.nuts.extend([square_nut, round_nut_1, round_nut_2])

        # Create arena
        mujoco_arena = PegsArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # More generous sampling ranges
        if self.placement_initializer is None:
            self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

            nut_sampler_specs = [
                ("SquareNutSampler", [-0.25, -0.05]),
                ("RoundNut1Sampler", [-0.05, 0.1]),
                ("RoundNut2Sampler", [0.1, 0.25]),
            ]
            for name, y_range in nut_sampler_specs:
                self.placement_initializer.append_sampler(
                    sampler=UniformRandomSampler(
                        name=name,
                        x_range=[-0.2, 0.2],   # wider x space
                        y_range=y_range,
                        rotation=[0, 0],       # fixed rotation for simplicity
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,  # disable to increase success
                        ensure_valid_placement=False,           # disable validation to avoid collision failures
                        reference_pos=self.table_offset,
                        z_offset=0.02,
                    )
                )

        # Reset before adding objects
        self.placement_initializer.reset()

        # Add nuts to samplers
        self.placement_initializer.add_objects_to_sampler("SquareNutSampler", square_nut)
        self.placement_initializer.add_objects_to_sampler("RoundNut1Sampler", round_nut_1)
        self.placement_initializer.add_objects_to_sampler("RoundNut2Sampler", round_nut_2)

        # Create model
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.nuts,
        )
