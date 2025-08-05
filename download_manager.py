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
    print("5. 📋 View available data files")
    print("6. 🧹 Clean up data directory")
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
            return None

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
                download_with_options(
                    downloader, 
                    downloader.download_sp500_data,
                    "S&P 500"
                )
                
            elif choice == "2":
                download_with_options(
                    downloader,
                    downloader.download_nasdaq_data,
                    "NASDAQ"
                )
                
            elif choice == "3":
                tickers = get_custom_tickers()
                if tickers:
                    download_with_options(
                        downloader,
                        lambda: downloader.download_custom_tickers(tickers),
                        "Custom Tickers"
                    )
                    
            elif choice == "4":
                print("\n🌍 Downloading both S&P 500 and NASDAQ...")
                print("⚠️  This is a large download and may take 30+ minutes!")
                confirm = input("🎯 Continue? (y/n): ").strip().lower()
                
                if confirm in ['y', 'yes']:
                    print("🚀 Starting comprehensive download...")
                    
                    # Download both
                    sp500_data = downloader.download_sp500_data()
                    nasdaq_data = downloader.download_nasdaq_data()
                    
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
                view_data_files(output_dir)
                
            elif choice == "6":
                clean_data_directory(output_dir)
                
            else:
                print("❌ Invalid choice. Please enter 0-6.")
                
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
