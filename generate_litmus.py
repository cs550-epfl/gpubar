import argparse
import numpy as np

def validate_gpu_count(value):
    ivalue = int(value)
    if not 2 <= ivalue <= 8:
        raise argparse.ArgumentTypeError(f"GPU count must be between 2 and 8, got {value}")
    return ivalue

def validate_execution_count(value):
    ivalue = int(value)
    if ivalue not in {1, 2}:
        raise argparse.ArgumentTypeError(f"Execution count must be either 1 or 2, got {value}")
    return ivalue

def validate_grid_size(value):
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"Grid size cannot be less than 1, got {value}")
    return ivalue

def validate_block_size(value, gpu_count):
    ivalue = int(value)
    if ivalue < gpu_count:
        raise argparse.ArgumentTypeError(f"Block size cannot be less than GPU count ({gpu_count}), got {value}")
    return ivalue

class SymbolGenerator:
    def __init__(self, grid_size: int, block_size: int):
        self.grid_size = grid_size
        self.block_size = block_size

    def __call__(self, gpu_id: int, bid: int, tid: int, base: str = 'x') -> str:
        global_tid = gpu_id * self.grid_size * self.block_size \
                   + bid * self.block_size \
                   + tid
        return f"{base}{global_tid}"  
  

class LitmusGenerator:
    def __init__(self, gpu_count: int, grid_size: int, block_size: int, executions: int):
        self.gpu_count : int = gpu_count
        self.grid_size : int = grid_size
        self.block_size: int = block_size
        self.executions: int = executions
        self.symbol_generator = SymbolGenerator(grid_size, block_size)

    def generate_header(self) -> list[str]:
        lines = []
        lines.append("PTX Multi-GPU Barrier Test")
        lines.append("\"dynamically generated\"")
        lines.append("{")

        # Pk:r0=1 for all threadspre_barrier_results =
        # Signals R{i}C{j}=1
        for i in range(self.gpu_count): # world size
            for j in range(self.gpu_count):
                lines.append(f"  R{i}C{j}=1;")

        for gid in range(self.gpu_count):
            for bid in range(self.grid_size):
                for tid in range(self.block_size):
                    lines.append(f"  {self.symbol_generator(gid, bid, tid)}=0;")

        for gid in range(self.gpu_count):
            for bid in range(self.grid_size):
                for tid in range(self.block_size):
                    lines.append(f"  {self.symbol_generator(gid, bid, tid, 'P')}:r1=0;")
            

        lines.append("}")
        lines.append("")
        return lines
    
    def generate_body(self) -> list[str]:
        # state
        cur_barrier_id = 0 # increase by 2 after each execution
        lines = []

        def generate_title(gpu_id: int, bid: int, tid: int) -> str:
            return [f"P{self.symbol_generator(gpu_id, bid, tid, '')}@cta {bid},gpu {gpu_id}"]
        
        def generate_pre_barrier(gpu_id: int, bid: int, tid: int) -> list[str]:
            lines = []
            lines.append(f"st.weak {self.symbol_generator(gpu_id, bid, tid)}, 1")
            lines.append(f"bar.cta.sync {cur_barrier_id}")
            return lines
            
        def generate_barrier(gpu_id: int, bid: int, tid: int) -> list[str]:
            lines = []
            lines.append(f"st.release.sys R{tid}C{gpu_id}, 0")
            lc_val = f"LC{self.symbol_generator(gpu_id, bid, tid, '')}{cur_barrier_id}"
            lines.append(f"{lc_val}:")
            lines.append(f"ld.acquire.sys r0, R{gpu_id}C{tid}")
            lines.append(f"bne r0, 0, {lc_val}")
            lines.append(f"bar.cta.sync {cur_barrier_id + 1}")
            return lines
        
        def generate_post_barrier(gpu_id: int, bid: int, tid: int):
            lines = []
            lc_val = f"LC{self.symbol_generator(gpu_id, bid, tid, '')}{cur_barrier_id + 1}"
            for gpu_id in range(self.gpu_count):
                for gid in range(self.grid_size):
                    for tid in range(self.block_size):
                        lines.append(f"ld.weak r2, {self.symbol_generator(gpu_id, gid, tid, 'x')}")
                        lines.append(f"bne r2, 1, {lc_val}")
                        lines.append(f"add r1, r1, 1")
            lines.append(f"{lc_val}:")
            return lines
        
        # title = " | ".join([generate_title(gid, bid, tid)
        #                     for gid in range(self.gpu_count) 
        #                     for bid in range(self.grid_size) 
        #                     for tid in range(self.block_size)]) + ";"
        # lines.append(title)

        title_results = []
        pre_barrier_results = []
        barrier_results = []
        post_barrier_results = []

        def pad_lines(lines: list[str], max_len: int) -> list[str]:
            return [line + " " * (max_len - len(line)) for line in lines]
        
        for gid in range(self.gpu_count):
            for bid in range(self.grid_size):
                for tid in range(self.block_size):
                    title_results.append(pad_lines(generate_title(gid, bid, tid), 30))
                    pre_barrier_results.append(pad_lines(generate_pre_barrier(gid, bid, tid), 30))
                    barrier_results.append(pad_lines(generate_barrier(gid, bid, tid), 30))
                    post_barrier_results.append(pad_lines(generate_post_barrier(gid, bid, tid), 30))

        title_transposed = np.array(title_results).T.tolist()
        pre_barrier_transposed = np.array(pre_barrier_results).T.tolist()
        barrier_transposed = np.array(barrier_results).T.tolist()
        post_barrier_transposed = np.array(post_barrier_results).T.tolist()        
        
        title_results = map(lambda x: " | ".join(x) + ";", title_transposed)
        lines.extend(title_results)
        pre_barrier_results = map(lambda x: " | ".join(x) + ";", pre_barrier_transposed)
        lines.extend(pre_barrier_results)
        barrier_results = map(lambda x: " | ".join(x) + ";", barrier_transposed)
        lines.extend(barrier_results)
        post_barrier_results = map(lambda x: " | ".join(x) + ";", post_barrier_transposed)
        lines.extend(post_barrier_results)

        return lines

    def generate_vc(self) -> list[str]:
        lines = []
        lines.append("forall")
        total_threads = self.gpu_count * self.grid_size * self.block_size
        condition = " /\ ".join([f"P{i}:r1=={total_threads}" for i in range(total_threads)])
        lines.append(f"({condition})")
        return lines

    def write(self):
        header = self.generate_header()
        body = self.generate_body()
        vc = self.generate_vc()

        with open("gen_barrier.litmus", "w") as f:
            f.write("\n".join(header + body + vc))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse GPU configuration parameters')
    
    parser.add_argument('--gpu-count', 
        type=validate_gpu_count,
        default=2,
        help='Number of GPUs connected by NVLink (2-8)'
    )
    parser.add_argument('--grid-size',
        type=validate_grid_size,
        default=1,
        help='Grid size'
    )
    parser.add_argument('--block-size',
        type=int,
        help='Block size'
    )
    parser.add_argument('--executions',
        type=validate_execution_count,
        default=1,
        help='Number of executions (1 or 2)'
    )
    args = parser.parse_args()
    
    # Validate block size after we know GPU count
    if args.block_size is None:
        args.block_size = args.gpu_count
    else:
        args.block_size = validate_block_size(args.block_size, args.gpu_count)

    print(f"Configuration:")
    print(f"GPUs: {args.gpu_count}")
    print(f"Grid size: {args.grid_size}")
    print(f"Block size: {args.block_size}")
    print(f"Number of executions: {args.executions}")

    generator = LitmusGenerator(args.gpu_count, args.grid_size, args.block_size, args.executions)
    generator.write()
