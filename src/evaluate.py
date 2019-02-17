from __future__ import absolute_import

from got10k.experiments import ExperimentOTB

from goturn import TrackerGOTURN


if __name__ == '__main__':
    # setup tracker
    net_path = "/home/abhinav.moudgil/pytorch_goturn.pth.tar"
    tracker = TrackerGOTURN(net_path=net_path)

    # setup experiments
    # got10k toolkit expects either extracted directories or zip files for
    # all sequences in OTB data directory
    experiments = [
        ExperimentOTB('/ssd_scratch/cvit/abhinav.m/OTB', version=2013)
        # ExperimentOTB('/ssd_scratch/cvit/abhinav.m/OTB', version=2015)
    ]

    # run tracking experiments and report performance
    for e in experiments:
        e.run(tracker, visualize=False)
        e.report([tracker.name])
