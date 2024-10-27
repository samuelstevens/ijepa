import logging

import beartype
import submitit
import tyro

import train

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


@beartype.beartype
def main(args: train.Args):
    if args.slurm:
        executor = submitit.SlurmExecutor(folder=args.log_to)
        executor.update_parameters(
            time=30,
            partition="preemptible",
            # partition="debug",
            # time=30 * 60,  # 30 hours
            # partition="gpu",
            gpus_per_node=args.gpus_per_node,
            ntasks_per_node=args.gpus_per_node,
            cpus_per_task=args.cpus_per_task,
            stderr_to_stdout=True,
            account=args.slurm_acct,
        )
    else:
        executor = submitit.DebugExecutor(folder=args.log_to)

    job = executor.submit(train.train, args)
    job.result()


if __name__ == "__main__":
    main(tyro.cli(train.Args))
