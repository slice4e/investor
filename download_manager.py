"""
Stock Data Download Manager

Interactive command-line tool for downloading comprehensive stock market data.
Supports S&P 500, NASDAQ, and custom ticker downloads with Excel/CSV export.
"""

from data_downloader import StockDataDownloader
import os

def display_menu():
    """Display the main menu options."""
    print("\n" + "="*60)
    print("📈 STOCK DATA DOWNLOAD MANAGER")
    print("="*60)
    print("1. 📊 Download S&P 500 stocks (all ~500 companies)")
    print("2. 🔬 Download NASDAQ stocks (NASDAQ-100)")
    print("3. 🎯 Download custom stock tickers")
    print("4. 🌍 Download both S&P 500 + NASDAQ")
    print("5. � Update existing data (incremental)")
    print("6. �📋 View data summary")
    print("7. 📂 View available data files")
    print("8. 🧹 Clean up data directory")
    print("0. ❌ Exit")
    print("="*60)

def get_custom_tickers():
    """Get custom ticker input from user."""
    print("\n📝 Enter stock tickers to download:")
    print("   Examples: AAPL, MSFT, GOOGL")
    print("   Separate multiple tickers with commas")
    
    while True:
        tickers_input = input("\n🎯 Enter tickers: ").strip()
        if not tickers_input:
            print("❌ Please enter at least one ticker")
            continue
        
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        tickers = [t for t in tickers if t]  # Remove empty strings
        
        if not tickers:
            print("❌ Please enter valid ticker symbols")
            continue
        
        print(f"\n📋 You entered: {', '.join(tickers)}")
        confirm = input("✅ Proceed with download? (y/n): ").strip().lower()
        
        if confirm in ['y', 'yes']:
            return tickers
        elif confirm in ['n', 'no']:
            print("Cancelled.")
def show_update_menu():
    """Display incremental update options."""
    print("\n" + "="*50)
    print("🔄 INCREMENTAL UPDATE OPTIONS")
    print("="*50)
    print("1. 📊 Update S&P 500 data")
    print("2. 🔬 Update NASDAQ data")
    print("3. 🎯 Update custom tickers")
    print("4. 🌍 Update all existing data")
    print("5. 📋 View update candidates")
    print("0. ⬅️ Back to main menu")
    print("="*50)

def handle_incremental_updates(downloader):
    """Handle incremental update operations."""
    while True:
        show_update_menu()
        choice = input("\n🎯 Choose update option (0-5): ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            print("\n🔄 Updating S&P 500 data...")
            result = downloader.update_existing_data(subfolder="sp500")
            print_update_summary(result, "S&P 500")
            
        elif choice == "2":
            print("\n🔄 Updating NASDAQ data...")
            result = downloader.update_existing_data(subfolder="nasdaq")
            print_update_summary(result, "NASDAQ")
            
        elif choice == "3":
            print("\n🔄 Updating custom tickers...")
            result = downloader.update_existing_data(subfolder="custom")
            print_update_summary(result, "Custom tickers")
            
        elif choice == "4":
            print("\n🔄 Updating all existing data...")
            print("⚠️  This will update ALL existing data across all categories!")
            confirm = input("🎯 Continue? (y/n): ").strip().lower()
            
            if confirm in ['y', 'yes']:
                results = downloader.update_all_data()
                for subfolder, result in results.items():
                    print_update_summary(result, subfolder)
            else:
                print("❌ Cancelled.")
                
        elif choice == "5":
            show_data_summary(downloader)
            
        else:
            print("❌ Invalid choice. Please enter 0-5.")
        
        if choice != "0":
            input("\n⏸️  Press Enter to continue...")

def print_update_summary(results, category):
    """Print summary of update results."""
    if not results:
        print(f"📭 No {category} data found to update.")
        return
    
    successful = sum(1 for status in results.values() if "✅" in status)
    updated = sum(1 for status in results.values() if "Updated" in status)
    current = sum(1 for status in results.values() if "current" in status)
    failed = len(results) - successful
    
    print(f"\n📊 {category} Update Summary:")
    print(f"   ✅ Total processed: {len(results)}")
    print(f"   🔄 Updated: {updated}")
    print(f"   📋 Already current: {current}")
    print(f"   ❌ Failed: {failed}")

def show_data_summary(downloader):
    """Show summary of all downloaded data."""
    summary_df = downloader.get_data_summary()
    
    if summary_df.empty:
        print("\n📭 No data found. Download some data first!")
        return
    
    print(f"\n📊 DATA SUMMARY ({len(summary_df)} tickers)")
    print("="*70)
    
    # Group by subfolder
    for subfolder in ['sp500', 'nasdaq', 'custom']:
        subset = summary_df[summary_df['Subfolder'] == subfolder]
        if not subset.empty:
            print(f"\n📁 {subfolder.upper()} ({len(subset)} tickers):")
            
            # Show stats
            recent_count = len(subset[subset['Days_Since_Update'] <= 1])
            old_count = len(subset[subset['Days_Since_Update'] > 7])
            
            print(f"   📋 Recent (≤1 day): {recent_count}")
            print(f"   ⚠️  Needs update (>7 days): {old_count}")
            
            # Show sample tickers
            sample_tickers = subset['Ticker'].head(5).tolist()
            print(f"   🎯 Sample: {', '.join(sample_tickers)}")
            if len(subset) > 5:
                print(f"   ... and {len(subset) - 5} more")

def download_with_incremental_option(downloader, download_type):
    """Download with option for incremental vs full download."""
    print(f"\n🚀 {download_type} Download Options:")
    print("1. 📥 Full download (complete history)")
    print("2. 🔄 Incremental download (update existing)")
    
    while True:
        choice = input("\n🎯 Choose download type (1-2): ").strip()
        
        if choice == "1":
            force_full = True
            break
        elif choice == "2":
            force_full = False
            break
        else:
            print("❌ Invalid choice. Please enter 1 or 2.")
    
    return force_full

def download_with_options(downloader, download_func, data_type):
    """Download data with export options."""
    print(f"\n🚀 Starting {data_type} download...")
    print("This may take several minutes depending on the number of stocks.")
    
    # Download data
    data = download_func()
    
    if not data:
        print("❌ No data was downloaded.")
        return
    
    success_count = len(data)
    print(f"\n✅ Successfully downloaded {success_count} stocks!")
    
    # Export options
    print(f"\n📂 Export Options:")
    print("1. 📄 CSV files only (individual files)")
    print("2. 📊 Excel file only (multi-sheet)")
    print("3. 📋 Both CSV and Excel")
    print("4. Skip export")
    
    while True:
        choice = input("\n🎯 Choose export option (1-4): ").strip()
        
        if choice == "1":
            print("💾 Saving as CSV files...")
            downloader.save_to_csv(data, data_type.lower().replace(" ", "_"))
            print("✅ CSV files saved!")
            break
        elif choice == "2":
            print("💾 Saving as Excel file...")
            filename = f"{data_type.lower().replace(' ', '_')}_data.xlsx"
            downloader.save_to_excel(data, filename)
            print("✅ Excel file saved!")
            break
        elif choice == "3":
            print("💾 Saving as both CSV and Excel...")
            downloader.save_to_csv(data, data_type.lower().replace(" ", "_"))
            filename = f"{data_type.lower().replace(' ', '_')}_data.xlsx"
            downloader.save_to_excel(data, filename)
            print("✅ Both formats saved!")
            break
        elif choice == "4":
            print("⏭️ Skipping export.")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")

def view_data_files(output_dir):
    """View available data files."""
    print(f"\n📁 Data files in '{output_dir}':")
    print("-" * 40)
    
    if not os.path.exists(output_dir):
        print("❌ Data directory doesn't exist yet.")
        return
    
    file_count = 0
    for root, dirs, files in os.walk(output_dir):
        if files:
            rel_path = os.path.relpath(root, output_dir)
            if rel_path != ".":
                print(f"\n📂 {rel_path}/")
            
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"   📄 {file} ({file_size:.1f} MB)")
                file_count += 1
    
    if file_count == 0:
        print("❌ No data files found.")
    else:
        print(f"\n📊 Total files: {file_count}")

def clean_data_directory(output_dir):
    """Clean up the data directory."""
    print(f"\n🧹 Clean Data Directory")
    print("-" * 30)
    
    if not os.path.exists(output_dir):
        print("❌ Data directory doesn't exist.")
        return
    
    # Count files
    file_count = 0
    for root, dirs, files in os.walk(output_dir):
        file_count += len(files)
    
    if file_count == 0:
        print("✅ Directory is already empty.")
        return
    
    print(f"⚠️  This will delete {file_count} files in '{output_dir}'")
    confirm = input("🗑️  Are you sure? (type 'DELETE' to confirm): ").strip()
    
    if confirm == "DELETE":
        import shutil
        shutil.rmtree(output_dir)
        print(f"🗑️  Directory '{output_dir}' has been deleted.")
    else:
        print("❌ Cancelled.")

def main():
    """Main interactive loop."""
    print("🚀 Welcome to the Stock Data Download Manager!")
    print("This tool downloads comprehensive historical stock data.")
    
    # Initialize downloader
    output_dir = "stock_market_data"
    downloader = StockDataDownloader(output_dir=output_dir)
    
    while True:
        display_menu()
        
        try:
            choice = input("\n🎯 Enter your choice (0-6): ").strip()
            
            if choice == "0":
                print("\n👋 Goodbye! Happy investing!")
                break
                
            elif choice == "1":
                # S&P 500 download
                force_full = download_with_incremental_option(downloader, "S&P 500")
                if force_full:
                    download_with_options(downloader, downloader.download_sp500_data, "S&P 500")
                else:
                    data = downloader.download_sp500_data_incremental(force_full=False)
                    print(f"✅ S&P 500 incremental update completed: {len(data)} stocks processed")
                
            elif choice == "2":
                # NASDAQ download
                force_full = download_with_incremental_option(downloader, "NASDAQ")
                if force_full:
                    download_with_options(downloader, downloader.download_nasdaq_data, "NASDAQ")
                else:
                    data = downloader.download_nasdaq_data_incremental(force_full=False)
                    print(f"✅ NASDAQ incremental update completed: {len(data)} stocks processed")
                
            elif choice == "3":
                # Custom tickers
                tickers = get_custom_tickers()
                if tickers:
                    force_full = download_with_incremental_option(downloader, "Custom Tickers")
                    data = downloader.download_custom_tickers_incremental(tickers, force_full=force_full)
                    print(f"✅ Custom ticker download completed: {len(data)} stocks processed")
                    
            elif choice == "4":
                print("\n🌍 Downloading both S&P 500 and NASDAQ...")
                print("⚠️  This is a large download and may take 30+ minutes!")
                confirm = input("🎯 Continue? (y/n): ").strip().lower()
                
                if confirm in ['y', 'yes']:
                    force_full = download_with_incremental_option(downloader, "Complete Market")
                    print("🚀 Starting comprehensive download...")
                    
                    if force_full:
                        # Full download
                        sp500_data = downloader.download_sp500_data()
                        nasdaq_data = downloader.download_nasdaq_data()
                    else:
                        # Incremental download
                        sp500_data = downloader.download_sp500_data_incremental(force_full=False)
                        nasdaq_data = downloader.download_nasdaq_data_incremental(force_full=False)
                    
                    # Combine and save
                    all_data = {**sp500_data, **nasdaq_data}
                    print(f"\n✅ Downloaded {len(all_data)} total stocks!")
                    
                    # Save combined
                    downloader.save_to_excel(all_data, "complete_market_data.xlsx")
                    combined_df = downloader.create_combined_dataset(all_data)
                    if not combined_df.empty:
                        combined_df.to_csv(f"{output_dir}/complete_market_data.csv", index=False)
                    
                    print("✅ Complete market data saved!")
                else:
                    print("❌ Cancelled.")
                    
            elif choice == "5":
                handle_incremental_updates(downloader)
                
            elif choice == "6":
                show_data_summary(downloader)
                
            elif choice == "7":
                view_data_files(output_dir)
                
            elif choice == "8":
                clean_data_directory(output_dir)
                
            else:
                print("❌ Invalid choice. Please enter 0-8.")
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Operation cancelled by user.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        # Pause before showing menu again
        if choice != "0":
            input("\n⏸️  Press Enter to continue...")
    
    # Final report
    downloader.generate_download_report()

if __name__ == "__main__":
    main()
