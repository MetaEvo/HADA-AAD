#!/bin/bash
# ============================================================================
# Setup Initial Generation - Run current code directly in Docker containers
# ============================================================================
#
# Usage:
#   bash setup_initial.sh                    # Run all domains
#   bash setup_initial.sh bbob_unconstrained # Run only bbob_unconstrained
#   bash setup_initial.sh bbob_constrained   # Run only bbob_constrained
#   bash setup_initial.sh metabox_mo         # Run only metabox_mo
#
# ============================================================================

REPO_NAME="HADA"
RUN_ID=$(date +%Y%m%d_%H%M%S_%N)

# Function to run initial eval for a domain in a container
run_initial_eval() {
    local domain=$1
    local output_dir="outputs/initial_${domain}_0"
    
    echo "=== Running initial eval for ${domain} ==="
    rm -rf "${output_dir}"
    mkdir -p "${output_dir}"
    
    CONTAINER_NAME="${REPO_NAME}-initial-${domain}-${RUN_ID}"
    docker create --name ${CONTAINER_NAME} --network=host -v "$(pwd)":/hada hada tail -f /dev/null
    docker start ${CONTAINER_NAME}
    
    echo "Running evaluation inside container..."
    docker exec -w /hada ${CONTAINER_NAME} bash -c "
        mkdir -p /hada/${output_dir}/${domain}_eval/agent_evals &&
        python -m domains.multi_seed_eval --domain ${domain} --output_dir /hada/${output_dir} 2>&1 | tee /hada/${output_dir}/generate.log
    "
    
    docker stop ${CONTAINER_NAME} 2>/dev/null || true
    docker rm ${CONTAINER_NAME} 2>/dev/null || true
    
    echo "=== Completed initial eval for ${domain} ==="
}

case "${1:-all}" in
    bbob_unconstrained)
        run_initial_eval "bbob_unconstrained"
        ;;
    bbob_constrained)
        run_initial_eval "bbob_constrained"
        ;;
    metabox_mo)
        run_initial_eval "metabox_mo"
        ;;
    all|"")
        run_initial_eval "bbob_unconstrained"
        run_initial_eval "bbob_constrained"
        run_initial_eval "metabox_mo"
        ;;
    *)
        echo "Unknown domain: $1"
        echo "Usage: bash setup_initial.sh [bbob_unconstrained|bbob_constrained|metabox_mo|all]"
        exit 1
        ;;
esac