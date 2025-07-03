#!/bin/bash

# TrainTicket Chaos Injection Script - REFACTORED FOR FLEXIBLE FAULT INJECTION
# This version allows selecting from a menu of 6 different fault types.
# FIXED: Improved containerName detection for StressChaos and ContainerKill to correctly target the main application container in multi-container pods (e.g., with Istio).

set -e

NAMESPACE="trainticket"
LOG_DIR="/tmp/chaos_logs"
TIMESTAMP=$(date "+%Y%m%d-%H%M%S")

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

mkdir -p $LOG_DIR

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}    TrainTicket Chaos Injection System${NC}"
echo -e "${BLUE}    Target Namespace: $NAMESPACE${NC}"
echo -e "${BLUE}    Execution Time: $(date)${NC}"
echo -e "${BLUE}================================================${NC}"

# Check environment for all required CRDs
check_environment() {
    echo -e "${YELLOW}Checking environment...${NC}"
    
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        echo -e "${RED}[FAIL] Namespace $NAMESPACE does not exist${NC}"; exit 1;
    fi
    echo -e "${GREEN}[OK] Namespace $NAMESPACE exists${NC}"
    
    if kubectl get crd networkchaos.chaos-mesh.org &> /dev/null; then
        echo -e "${GREEN}[OK] NetworkChaos CRD available${NC}";
    else
        echo -e "${RED}[FAIL] NetworkChaos CRD not available. Please install Chaos Mesh.${NC}"; exit 1;
    fi
    
    if kubectl get crd stresschaos.chaos-mesh.org &> /dev/null; then
        echo -e "${GREEN}[OK] StressChaos CRD available${NC}";
    else
        echo -e "${RED}[FAIL] StressChaos CRD not available. Please install Chaos Mesh.${NC}"; exit 1;
    fi

    if kubectl get crd podchaos.chaos-mesh.org &> /dev/null; then
        echo -e "${GREEN}[OK] PodChaos CRD available${NC}";
    else
        echo -e "${RED}[FAIL] PodChaos CRD not available. Please install Chaos Mesh.${NC}"; exit 1;
    fi
    
    echo ""
}

# Get clean service list
get_service_list() {
    kubectl get pods -n $NAMESPACE -o jsonpath='{.items[*].metadata.labels.app}' 2>/dev/null | \
    tr ' ' '\n' | sort | uniq | grep -v '^$' | grep -v 'Service List' | head -10
}

# Select target service
select_target_service() {
    echo -e "${YELLOW}Available Service Labels:${NC}" >&2
    readarray -t services < <(get_service_list)
    
    for i in "${!services[@]}"; do
        echo "$((i+1)). ${services[$i]}" >&2
    done
    echo "" >&2
    
    read -p "Please select target service (1-${#services[@]}): " choice
    local selected_service=""
    
    if [[ $choice -gt 0 && $choice -le ${#services[@]} ]]; then
        selected_service="${services[$((choice-1))]}"
    else
        echo -e "${RED}Invalid selection. Exiting.${NC}" >&2
        exit 1
    fi
    
    echo "$selected_service"
}

# === FAULT INJECTION FUNCTIONS ===

# 1. CPU Load
create_cpu_load_chaos() {
    local target_service="$1"
    local cpu_load="${2:-80}"
    local duration="${3:-3m}"
    
    # VVVVVV 修正：更精确地自动检测主应用容器名 VVVVVV
    local container_name=$(kubectl get pod -n $NAMESPACE -l app=$target_service -o jsonpath='{.items[0].spec.containers[?(@.name!="istio-proxy")].name}')
    if [[ -z "$container_name" ]]; then
        echo -e "${RED}[FAIL] Could not automatically determine a target container for service $target_service.${NC}"
        exit 1
    fi
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    echo -e "${RED}=== Injecting CPU Load Chaos ===${NC}"
    echo -e "${YELLOW}Target Service: $target_service, Target Container: $container_name, Load: $cpu_load%, Duration: $duration${NC}"
    
    kubectl apply -f - <<EOF
apiVersion: chaos-mesh.org/v1alpha1
kind: StressChaos
metadata:
  name: cpu-load-$TIMESTAMP
  namespace: $NAMESPACE
spec:
  mode: all
  selector:
    labelSelectors:
      app: "$target_service"
  # VVVVVV 修正：在spec中指定正确的目标容器 VVVVVV
  containerNames:
  - "$container_name"
  # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  stressors:
    cpu:
      workers: 2
      load: $cpu_load
  duration: "$duration"
EOF
    echo -e "${GREEN}[OK] CPU Load chaos created${NC}"
}

# 2. Memory Stress
create_memory_stress_chaos() {
    local target_service="$1"
    local memory_size="${2:-512MB}"
    local duration="${3:-4m}"
    
    # VVVVVV 修正：更精确地自动检测主应用容器名 VVVVVV
    local container_name=$(kubectl get pod -n $NAMESPACE -l app=$target_service -o jsonpath='{.items[0].spec.containers[?(@.name!="istio-proxy")].name}')
    if [[ -z "$container_name" ]]; then
        echo -e "${RED}[FAIL] Could not automatically determine a target container for service $target_service.${NC}"
        exit 1
    fi
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    echo -e "${RED}=== Injecting Memory Stress Chaos ===${NC}"
    echo -e "${YELLOW}Target Service: $target_service, Target Container: $container_name, Size: $memory_size, Duration: $duration${NC}"
    
    kubectl apply -f - <<EOF
apiVersion: chaos-mesh.org/v1alpha1
kind: StressChaos
metadata:
  name: memory-stress-$TIMESTAMP
  namespace: $NAMESPACE
spec:
  mode: all
  selector:
    labelSelectors:
      app: "$target_service"
  # VVVVVV 修正：在spec中指定正确的目标容器 VVVVVV
  containerNames:
  - "$container_name"
  # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  stressors:
    memory:
      workers: 1
      size: "$memory_size"
  duration: "$duration"
EOF
    echo -e "${GREEN}[OK] Memory Stress chaos created${NC}"
}

# 3. Network Delay
create_network_delay_chaos() {
    local target_service="$1"
    local delay_ms="${2:-500}"
    local duration="${3:-5m}"
    
    echo -e "${RED}=== Injecting Network Delay Chaos ===${NC}"
    echo -e "${YELLOW}Target: $target_service, Delay: ${delay_ms}ms, Duration: $duration${NC}"
    
    kubectl apply -f - <<EOF
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: network-delay-$TIMESTAMP
  namespace: $NAMESPACE
spec:
  action: delay
  mode: all
  selector:
    labelSelectors:
      app: "$target_service"
  delay:
    latency: "${delay_ms}ms"
    correlation: "100"
    jitter: "10ms"
  duration: "$duration"
EOF
    echo -e "${GREEN}[OK] Network Delay chaos created${NC}"
}

# 4. Network Loss
create_network_loss_chaos() {
    local target_service="$1"
    local loss_rate="${2:-20}" # Default loss rate 20%
    local duration="${3:-3m}"
    
    echo -e "${RED}=== Injecting Network Loss Chaos ===${NC}"
    echo -e "${YELLOW}Target: $target_service, Loss Rate: ${loss_rate}%, Duration: $duration${NC}"

    kubectl apply -f - <<EOF
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: network-loss-$TIMESTAMP
  namespace: $NAMESPACE
spec:
  action: loss
  mode: all
  selector:
    labelSelectors:
      app: "$target_service"
  loss:
    loss: "${loss_rate}"
    correlation: "100"
  duration: "$duration"
EOF
    echo -e "${GREEN}[OK] Network Loss chaos created${NC}"
}

# 5. Pod Failure
create_pod_failure_chaos() {
    local target_service="$1"
    local duration="${2:-5m}"

    echo -e "${RED}=== Injecting Pod Failure Chaos ===${NC}"
    echo -e "${YELLOW}Target: $target_service, Duration: $duration${NC}"
    echo -e "${YELLOW}This will repeatedly kill the pod for the specified duration.${NC}"

    kubectl apply -f - <<EOF
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: pod-failure-$TIMESTAMP
  namespace: $NAMESPACE
spec:
  action: pod-failure
  mode: all
  selector:
    labelSelectors:
      app: "$target_service"
  duration: "$duration"
EOF
    echo -e "${GREEN}[OK] Pod Failure chaos created${NC}"
}

# 6. Container Kill
create_container_kill_chaos() {
    local target_service="$1"
    local duration="${2:-5m}"

    # VVVVVV 修正：更精确地自动检测主应用容器名 VVVVVV
    local container_name=$(kubectl get pod -n $NAMESPACE -l app=$target_service -o jsonpath='{.items[0].spec.containers[?(@.name!="istio-proxy")].name}')
    
    if [[ -z "$container_name" ]]; then
        echo -e "${RED}[FAIL] Could not automatically determine a target container for service $target_service.${NC}"
        exit 1
    fi
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    echo -e "${RED}=== Injecting Container Kill Chaos ===${NC}"
    echo -e "${YELLOW}Target Service: $target_service${NC}"
    echo -e "${YELLOW}Target Container: $container_name (auto-detected)${NC}"
    echo -e "${YELLOW}Duration: $duration${NC}"

    kubectl apply -f - <<EOF
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: container-kill-$TIMESTAMP
  namespace: $NAMESPACE
spec:
  action: container-kill
  mode: all
  containerNames:
  - "$container_name"
  selector:
    labelSelectors:
      app: "$target_service"
  duration: "$duration"
EOF
    echo -e "${GREEN}[OK] Container Kill chaos created${NC}"
}


# === MAIN EXECUTION LOGIC ===
main() {
    check_environment
    
    echo "Step 1: Select target service"
    target_service=$(select_target_service)
    echo -e "${BLUE}Target service selected: '$target_service'${NC}"
    echo ""

    echo "Step 2: Select fault type to inject"
    echo -e "${YELLOW}Available Fault Types:${NC}"
    echo "1. cpu_load"
    echo "2. memory_stress"
    echo "3. network_delay"
    echo "4. network_loss"
    echo "5. pod_failure"
    echo "6. container_kill"
    echo ""
    read -p "Please select a fault type (1-6): " fault_choice

    echo ""

    case "$fault_choice" in
        1) create_cpu_load_chaos "$target_service" ;;
        2) create_memory_stress_chaos "$target_service" ;;
        3) create_network_delay_chaos "$target_service" ;;
        4) create_network_loss_chaos "$target_service" ;;
        5) create_pod_failure_chaos "$target_service" ;;
        6) create_container_kill_chaos "$target_service" ;;
        *) echo -e "${RED}Invalid selection. Exiting.${NC}"; exit 1 ;;
    esac

    echo ""
    echo -e "${GREEN}=== Chaos Injection Experiment Completed! ===${NC}"
    echo "You can monitor the status using K9s or 'kubectl get pods -n $NAMESPACE -w'"
    echo "To clean up, run: kubectl delete networkchaos,stresschaos,podchaos --all -n $NAMESPACE"
}

# Command line argument handling
case "${1:-}" in
    "cleanup")
        echo "Cleaning up all chaos resources in namespace $NAMESPACE..."
        kubectl delete networkchaos,stresschaos,podchaos --all -n $NAMESPACE
        ;;
    *)
        main
        ;;
esac