#!/usr/bin/env python3
"""Run automated paper trading until profit target is reached."""

import logging
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from firm.brokers.ibkr import IBKRBroker
from firm.logging_setup import setup_logging

setup_logging(log_file="data/logs/run_until_profitable.log")
log = logging.getLogger(__name__)


def run_until_profitable(
    profit_target_usd: float = 100.0,
    max_cycles: int = 100,
    cycle_interval_seconds: int = 60,
):
    """Run paper trading until profit target or max cycles reached.
    
    Args:
        profit_target_usd: Stop when account profit exceeds this amount
        max_cycles: Maximum number of trading cycles to run
        cycle_interval_seconds: Delay between cycles (seconds)
    """
    
    # Connect to IB Gateway
    log.info(f"Connecting to IB Gateway at {os.getenv('IBKR_HOST', '127.0.0.1')}:{os.getenv('IBKR_PAPER_PORT', '4002')}")
    broker = IBKRBroker(
        host=os.getenv("IBKR_HOST", "127.0.0.1"),
        port=int(os.getenv("IBKR_PAPER_PORT", "4002")),
        client_id=int(os.getenv("IBKR_CLIENT_ID", "1")),
    )
    
    try:
        broker.connect()
        log.info("✅ Connected to IB Gateway")
    except Exception as e:
        log.error(f"❌ Failed to connect to IB Gateway: {e}")
        return False
    
    # Get initial account balance
    initial_account = broker.get_account()
    initial_equity = initial_account.get("equity", 0)
    log.info(f"Initial account equity: ${initial_equity:,.2f}")
    log.info(f"Available buying power: ${initial_account.get('buying_power', 0):,.2f}")
    
    cycle_count = 0
    final_equity = initial_equity
    try:
        while cycle_count < max_cycles:
            cycle_count += 1
            
            try:
                # Check current account
                current_account = broker.get_account()
                current_equity = current_account.get("equity", 0)
                final_equity = current_equity
                current_profit = current_equity - initial_equity
                
                log.info(f"\n--- Cycle {cycle_count}/{max_cycles} ---")
                log.info(f"Equity: ${current_equity:,.2f} | Profit/Loss: ${current_profit:,.2f}")
                log.info(f"Buying Power: ${current_account.get('buying_power', 0):,.2f}")
                
                # Check if profit target reached
                if current_profit >= profit_target_usd:
                    log.info(f"\n✅ SUCCESS! Profit target of ${profit_target_usd:,.2f} reached!")
                    log.info(f"Final profit: ${current_profit:,.2f}")
                    broker.disconnect()
                    return True
                
                # Get positions
                positions = broker.get_positions()
                if positions:
                    log.info(f"📊 Open positions: {len(positions)}")
                    for pos in positions:
                        pnl = (pos.market_value - pos.quantity * pos.avg_cost)
                        log.info(f"  {pos.symbol}: {pos.quantity:.0f}x @ ${pos.avg_cost:.2f} (${pnl:,.0f} P&L)")
                else:
                    log.info("📊 Open positions: 0")
                
                # NOTE: For full strategy execution, use the API server instead:
                #   firm-api
                # Then POST to /live/start with:
                #   {"broker": "ibkr_paper", "schedule": "market_open", "approval_mode": "auto"}
                
                # Wait before next cycle
                if cycle_count < max_cycles:
                    log.info(f"⏳ Waiting {cycle_interval_seconds}s before next cycle...")
                    time.sleep(cycle_interval_seconds)
            
            except Exception as e:
                log.error(f"❌ Cycle error: {e}", exc_info=True)
                continue
    
    except KeyboardInterrupt:
        log.info("\n⏹️  Trading stopped by user")
    except Exception as e:
        log.error(f"❌ Error during trading: {e}", exc_info=True)
        return False
    finally:
        try:
            if broker.is_connected():
                broker.disconnect()
                log.info("Disconnected from IB Gateway")
        except Exception as e:
            log.error(f"Error disconnecting: {e}")
    
    # Final summary
    log.info("\n=== Final Summary ===")
    log.info(f"Cycles executed: {cycle_count}/{max_cycles}")
    log.info(f"Initial equity: ${initial_equity:,.2f}")
    log.info(f"Final equity: ${final_equity:,.2f}")
    final_profit = final_equity - initial_equity
    log.info(f"Total profit/loss: ${final_profit:,.2f}")
    
    return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run automated paper trading until profitable")
    parser.add_argument("--profit-target", type=float, default=100.0, help="Profit target in USD (default: 100)")
    parser.add_argument("--max-cycles", type=int, default=100, help="Maximum trading cycles (default: 100)")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between cycles (default: 60)")
    
    args = parser.parse_args()
    
    success = run_until_profitable(
        profit_target_usd=args.profit_target,
        max_cycles=args.max_cycles,
        cycle_interval_seconds=args.interval,
    )
    
    sys.exit(0 if success else 1)
