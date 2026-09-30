# Clients newly excluded by the NAV blocked flag

The old lookup saw blocked-ness only as `***` in a client name (56 clients).
NAV has a real column. Clients flagged there that traded in the last 12 months are
now excluded from every vendor queue. Verify with the business before
treating this as settled.

- newly flagged, with trade in the last 12 months: **222 clients**
- blocking reasons seen: unpaid invoices, unsettled balances, manual block, management decision

> The per-client breakdown (names, reason, trailing revenue) was removed from this
> public copy: it identified real trading partners and their payment status.
> Re-running `src/contact_queue.py` regenerates it locally under `data/`.
