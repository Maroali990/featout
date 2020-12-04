import torch
import torchvision
import torchvision.transforms.functional as TF
import numpy as np
import os

from featout.utils.utils import get_max_activation
from featout.interpret import simple_gradient_saliency
from featout.utils.blur import zero_out, blur_around_max
from captum.attr import visualization as viz
from featout.utils.plotting import plot_together


# Inherit from any pytorch dataset class
def get_featout_dataset(dataset, *args, **kwargs):

    class Featout(dataset):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # initial stage: no blurring
            self.featout = False
            self.plotting = "outputs"

        def __getitem__(self, index):
            image, label = super().__getitem__(index)

            if self.featout:
                in_img = torch.unsqueeze(image, 0)

                # TODO: batch the whole gradient computing? would be a speed up
                _, predicted_lab = torch.max(
                    self.featout_model(in_img).data, 1
                )
                # only do featout if it was predicted correctly
                if predicted_lab == label:
                    gradients = torch.squeeze(
                        self.algorithm(self.featout_model, in_img, label)
                    ).numpy()
                    # Compute point of maximum activation
                    max_x, max_y = get_max_activation(gradients)

                    # blurr out and write into image variable
                    blurred_image = self.blur_method(
                        in_img, (max_x, max_y), patch_radius=4
                    )
                    # TODO: test by saving the image before and after
                    if self.plotting is not None:
                        new_grads = torch.squeeze(
                            self.algorithm(
                                self.featout_model, blurred_image, label
                            )
                        ).numpy()
                        plot_together(
                            image,
                            gradients,
                            blurred_image[0],
                            new_grads,
                            save_path=os.path.join(
                                self.plotting, f"images_{index}.png"
                            )
                        )

                    image = torch.squeeze(blurred_image)

            return image, label

        def start_featout(
            self,
            model,
            blur_method=blur_around_max,
            algorithm=simple_gradient_saliency
        ):
            """
            We can set here whether we want to blur or zero and what gradient alg
            """
            # TODO: pass predicted labels because we only do featout if it is
            # predicted correctly
            print("start featout")
            self.featout = True
            self.algorithm = simple_gradient_saliency
            self.featout_model = model
            self.blur_method = blur_method
            self.gradient_algorithm = algorithm

        def stop_featout(self, ):
            self.featout = False

    return Featout(*args, **kwargs)


# Inspired from https://discuss.pytorch.org/t/changing-transformation-applied-to-data-during-training/15671/3
# Example usage:

# dataset = Featout(normal arguments of super dataset)
# loader = DataLoader(dataset, batch_size=2, num_workers=2, shuffle=True)
# loader.dataset.start_featout(net)
