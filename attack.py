## attack.py imports
import foolbox as fb

## Return the PGD attack 

def return_attack(attack_key, pgd_steps):
    ## Load corresponding attack based on given argument
    if attack_key is None:
        pass
    elif attack_key == 'FGSM':
        attack = fb.attacks.FGSM()
    elif attack_key == 'L2PGD':
        attack = fb.attacks.L2PGD(steps=pgd_steps)
    elif attack_key == 'LinfPGD':
        attack = fb.attacks.LinfPGD(steps=pgd_steps)
    else:
        raise ValueError(f'Attack {attack} not known')

    return attack
