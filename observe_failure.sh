# bash observe_failure.sh <hf-token>
# python examples/scripts/diffusion_dpo.py --hf-user-access-token ${1}
accelerate launch examples/scripts/diffusion_dpo.py --hf-user-access-token ${1}
