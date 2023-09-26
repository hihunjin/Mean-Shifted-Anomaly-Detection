[
    '5_o_Clock_Shadow',
    'Arched_Eyebrows',
    'Attractive',
    'Bags_Under_Eyes',
    'Bald',
    'Bangs',
    'Big_Lips',
    'Big_Nose',
    'Black_Hair',
    'Blond_Hair',
    'Blurry',
    'Brown_Hair',
    'Bushy_Eyebrows',
    'Chubby',
    'Double_Chin',
    'Eyeglasses', #
    'Goatee',
    'Gray_Hair',
    'Heavy_Makeup',
    'High_Cheekbones',
    'Male',
    'Mouth_Slightly_Open',
    'Mustache', #
    'Narrow_Eyes', 
    'No_Beard',
    'Oval_Face',
    'Pale_Skin',
    'Pointy_Nose',
    'Receding_Hairline',
    'Rosy_Cheeks',
    'Sideburns',
    'Smiling',  #
    'Straight_Hair', 
    'Wavy_Hair',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'Young' #
] 

from datasets.celeba import celeba
from utils import make_condition, make_subset


dataset = celeba()
train_condition_config = {
    dataset.attr_names[15]: True,  # Eyeglasses
    dataset.attr_names[39]: True,  # Young

    dataset.attr_names[22]: True,  # Mustache
    dataset.attr_names[31]: True,  # Smiling
}
print(dataset.attr_names)
train_condition = make_condition(dataset.attr_names, train_condition_config)
dataset = make_subset(dataset, train_condition)

for data in dataset:
    print(data[1])
    len
