<?php

require_once '../db.php';



@session_start();
if(!isset($_SESSION['is_login']) || !$_SESSION['is_login'])
{
	//直接轉跳到 index.php
	header("Location: /final/list.php");
}
require_once '../login_status.php';
?>
<?php



?>
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
        <meta name="description" content="" />
        <meta name="author" content="" />
        <title>購物車</title>
        <link rel="icon" type="image/x-icon" href="assets/favicon.ico" />
        <script src="https://use.fontawesome.com/releases/v5.15.3/js/all.js" crossorigin="anonymous"></script>
        <link href="final.css" rel="stylesheet" />
    </head>
    <body>
        
        <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <div class="container">
                <a class="navbar-brand" href="final2.php">YAPPI Shop</a>
                <button class="navbar-toggler navbar-toggler-right" type="button" data-toggle="collapse" data-target="#navbarResponsive" aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation"><span class="navbar-toggler-icon"></span></button>
                <div class="collapse navbar-collapse" id="navbarResponsive">
                    <ul class="navbar-nav ml-auto">
                        <!-- <li class="nav-item"><a class="nav-link" href="cartview2.php">購物車</a></li> -->
                        <li class="nav-item"><a class="nav-link" href="final2.php">繼續逛</a></li>
                        <li class="nav-item"><a class="nav-link" href="contact.html">聊天室</a></li>
                        
                    </ul>
                </div>
            </div>
        </nav>
        
        <section class="py-5">
            <div class="container">
           
                <h1>
                    您的購物車
                    <small>購購購~</small>
                </h1>
                <ol class="breadcrumb mb-4">
                    <li class="breadcrumb-item"><a href="final2.php">首頁</a></li>
                    <li class="breadcrumb-item active">訂單詳情</li>
                </ol>
                <div class="row">
                    
                <?php
				$i=0;
				$where_str='';
				$per_page = 9; //每頁最多9筆，這數字可以自由更改
				if(isset($_GET['page'])){
					$page=$_GET['page'];
				}	else{
					$page=1;
				}
				$start = ($page-1)*$per_page;
				$where_str='';
				?>

                <?php

				$link = mysqli_connect($host,$dbuser,$dbpw,$dbname) or die ("-1");
				  mysqli_select_db($link,$dbname) or die("cannot connect to db");
				  $query = "SELECT * FROM commodity $where_str ORDER BY c_id DESC ";
				  $result = mysqli_query($link,$query) or die("cannot connect to table" . mysqli_error( ));
				  $I=0;
				  while ( $row = mysqli_fetch_array($result) ) {
					$I++;
				  }
				  $total=$I;
				  $pages=ceil($total/$per_page);
				  if($page>$pages)
				  {
					  $page=$pages;
				  }
				  $start = ($page-1)*$per_page;
				?>

<?php
                $pdo=new PDO('mysql:host=localhost;dbname=final;charset=utf8','root','');    //連結資料庫，用PDO
                if (isset($_REQUEST['command1'])){
                    switch($_REQUEST['command1']){
                        case 'delete':
                            $sql=$pdo->prepare('DELETE FROM cart WHERE thing_id=?');
                            $sql->execute([$_REQUEST['id']]);
                            break;
                    }
                }
                if (isset($_REQUEST['command'])){
                    switch($_REQUEST['command']){
                        case 'delete':
                            $sql = $pdo->prepare('DELETE FROM cart');
                            $sql->execute();
                            break;
                        // case 'update':
                        //     $sql=$pdo->prepare('UPDATE commodity SET quantity-=? WHERE c_id=?');
                        //     // $amount=$pdo->prepare('SELECT quantity FROM commodity WHERE c_id=?');
                        //     $sql->execute($_REQUEST['amount'],$_REQUEST['id']);
                            
                        //     // $sql=$pdo->prepare('DELETE FROM cart WHERE thing_id=?');
                        //     // $sql->execute([$_REQUEST['id']]);
                        //     // $sql = "UPDATE `commodity` SET `quantity` =`quantity`-'{$row['amount']}' WHERE `c_id`='{$row['thing_id']}';";
                        //     break;
                    }
                }

                // foreach($pdo->query('select * from cart') as $row){
                //     $sql = "UPDATE `commodity` SET `quantity` =`quantity`-'{$row['amount']}' WHERE `c_id`='{$row['thing_id']}';";
                //     $query = mysqli_query($_SESSION['link'], $sql);
                //     }
                //     header("Location:final2.php");
                $total=0;
               
                    

                foreach($pdo->query('select * from cart') as $row){
                ?>

                            <div class="im"><?php echo '<p><img src="',$row['url'],'"></p>';?></div>  <!--圖片大小待調-->
                            <div class="col-lg-3 col-md-4 col-sm-6 mb-4">
                            <div class="card h-100">
                            <div class="card-body">
                                <h4 class="card-title"><a href="commodity.php?c_id=<?php echo $row['thing_id']; ?>" ><?php echo $row['name']; ?></a></h4>
                                <p class="card-text">商品價格: <?php echo $row['price']; ?> <br> 商品數量: <?php echo $row['amount']; ?><br>小計: <?php $subtotal=$row['price']*$row['amount']; echo $subtotal; ?> </p>
                                <?php $total+=$subtotal; ?>
                    
                                <form method="post" class="delete">
                                <input type="hidden" name="command1" value="delete">
                                <?php 
								// $sql = "UPDATE `commodity` SET `quantity` =`quantity`-'{$row['amount']}' WHERE `c_id`='{$row['thing_id']}';";
								// $query = mysqli_query($_SESSION['link'], $sql);
                                echo'<input type="hidden" name="id" value="',$row['thing_id'],'">'?>
                                <input type="submit" value="刪除">
                                </form>
                            </div>
                        </div>
                    </div>
                   
               
                <?php 
				} //foreach
				?>
                 <section class="py-5">
            </div>
            <div class="container">
           
                <h2>
                合計:<?php echo $total; ?>元
                </h2>
                </div>
            <hr>
            <div class="q">
            <h1><i class="fas fa-cart-arrow-down"></i> 結帳</h1>
            <h2>付款方式</h2>
            <form id="form" class="checkout" method="post">
            
            <div class="check">
            <input type="radio" name="method" value="1" id="cash" checked="checked" > 貨到付款 <i class="fas fa-wallet"></i><br>
            <input type="radio" name="method" value="1" id="creditcard" > 信用卡/金融卡 <i class="far fa-credit-card"></i><br>
            <input type="radio" name="method" value="1" id="ATM"> 銀行轉帳 <i class="fas fa-exchange-alt"></i><br>
            </div>
            <br>
            <input type="hidden" name="command" value="delete">
            <?php echo'<input type="hidden" name="id" value="',$row['thing_id'],'">'?>
            <?php echo'<input type="hidden" name="amount" value="',$row['amount'],'">'?>
            <input type='submit' value="下訂單" style="width:1120px; height: 40px; font-size:20px; color: #fff ;background-color: #6c757d; border-radius: 0.25rem;" ><br>
            </form>
            </div>
       
        
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/js/bootstrap.bundle.min.js"></script>
		  <script src="https://code.jquery.com/jquery-1.12.4.min.js"></script>
   
    <script>
     $(document).on("ready", function() {
				
				
				
				$("form.checkout").on("submit", function(){
				alert("購買成功");
				
				
				 window.location.href = "checkout.php"; 
				 
      
				});
      });

	  </script>
	  
   <!-- <script>
// $("#form").submit(function(e) {
//     var method =$("input[name='method']:checked").val(); 
//    if( typeof(method) == "undefined"){
//     Swal.fire({
//       icon: 'warning',
//       title: 'Oops...',
//       text: '請選擇付款方式',
//     });
//     return;
//   } else {
//     var form = $(this);
//     var url = form.attr('action');
//     $.ajax({
//       type: "POST",
//       url: url,  
//       data: ,
//       success function(data){
//        Swal.fire({
//             icon: 'success',
//             title: 'OK',
//             text: '購買成功',
//             allowOutsideClick: false,
//             showCancelButton: false,
//           })}
    

       
//         }
//       }
//     });
//     e.preventDefault(); // avoid to execute the actual submit of the form.
//   }
// });
// </script>   -->
 </body>
</html>
<style type="text/css">
.check{
    font-weight:bold;
    font-size: x-large;
    margin-left: 10px;

}
</style>

