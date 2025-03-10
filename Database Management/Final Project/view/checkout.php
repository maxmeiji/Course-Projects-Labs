 <?php 
 
 require_once '../db.php';
@session_start();
    $pdo=new PDO('mysql:host=localhost;dbname=final;charset=utf8','root','');    //連結資料庫，用PDO
    if (isset($_REQUEST['command'])){
    switch($_REQUEST['command']){
    case 'delete':
    $sql=$pdo->prepare('DELETE FROM cart WHERE thing_id=?');
     $sql->execute([$_REQUEST['id']]);
     break;
         }
}

	foreach($pdo->query('select * from cart') as $row){
	$sql = "UPDATE `commodity` SET `quantity` =`quantity`-'{$row['amount']}' WHERE `c_id`='{$row['thing_id']}';";
	$query = mysqli_query($_SESSION['link'], $sql);
	
	}	
	header("Location:final2.php");	
?>
